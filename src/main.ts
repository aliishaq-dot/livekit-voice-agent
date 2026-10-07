import { AutoSubscribe, ServerOptions, cli, defineAgent, inference } from '@livekit/agents';
import { EnhancerModel, audioEnhancement } from '@livekit/plugins-ai-coustics';
import {
  AudioStream,
  type LocalParticipant,
  ParticipantKind,
  type RemoteParticipant,
  type RemoteTrack,
  type RemoteTrackPublication,
  RoomEvent,
  TrackSource,
} from '@livekit/rtc-node';
import dotenv from 'dotenv';
import { fileURLToPath } from 'node:url';

dotenv.config({ path: '.env.local' });

const INTERIM_TRANSCRIPT = 1;
const FINAL_TRANSCRIPT = 2;

export default defineAgent({
  entry: async (ctx) => {
    const workers = new Map<string, TranscriptionWorker>();
    console.info('[captions] agent entrypoint started');

    const startTrack = (
      track: RemoteTrack,
      publication: RemoteTrackPublication,
      participant: RemoteParticipant,
    ) => {
      console.info('[captions] track subscribed', {
        participantIdentity: participant.identity,
        participantKind: participant.kind,
        trackSid: publication.sid,
        source: publication.source,
      });

      if (
        participant.kind === ParticipantKind.AGENT ||
        publication.source !== TrackSource.SOURCE_MICROPHONE ||
        !publication.sid ||
        workers.has(publication.sid)
      ) {
        console.info('[captions] track ignored', {
          participantIdentity: participant.identity,
          trackSid: publication.sid,
          reason:
            participant.kind === ParticipantKind.AGENT
              ? 'agent participant'
              : publication.source !== TrackSource.SOURCE_MICROPHONE
                ? 'not a microphone track'
                : !publication.sid
                  ? 'missing track SID'
                  : 'worker already exists',
        });
        return;
      }

      const trackSid = publication.sid;
      const localParticipant = ctx.room.localParticipant;
      if (!localParticipant) {
        throw new Error(
          `Agent connected without a local participant while starting track ${trackSid}`,
        );
      }

      const worker = new TranscriptionWorker(
        localParticipant,
        participant.identity,
        trackSid,
        track,
      );
      workers.set(trackSid, worker);
      console.info('[captions] starting transcription worker', {
        participantIdentity: participant.identity,
        trackSid,
        workerCount: workers.size,
      });
      void worker.start().catch((error: unknown) => {
        console.error(`Transcription failed for participant ${participant.identity}:`, error);
        workers.delete(trackSid);
        worker.stop();
      });
    };

    const stopTrack = (_track: RemoteTrack, publication: RemoteTrackPublication) => {
      console.info('[captions] track unsubscribed', {
        trackSid: publication.sid,
      });
      if (!publication.sid) {
        console.warn('[captions] cannot stop worker: track SID is missing');
        return;
      }
      const worker = workers.get(publication.sid);
      if (!worker) {
        console.warn('[captions] no worker found for unsubscribed track', {
          trackSid: publication.sid,
        });
        return;
      }
      workers.delete(publication.sid);
      worker.stop();
      console.info('[captions] transcription worker stopped', {
        trackSid: publication.sid,
        workerCount: workers.size,
      });
    };

    const stopParticipant = (participant: RemoteParticipant) => {
      console.info('[captions] participant disconnected', {
        participantIdentity: participant.identity,
      });
      for (const [trackSid, worker] of workers) {
        if (worker.participantIdentity === participant.identity) {
          workers.delete(trackSid);
          worker.stop();
          console.info('[captions] stopped participant worker', {
            participantIdentity: participant.identity,
            trackSid,
            workerCount: workers.size,
          });
        }
      }
    };

    const trackParticipant = async (_ctx: typeof ctx, participant: RemoteParticipant) => {
      console.info('[captions] participant entrypoint', {
        participantIdentity: participant.identity,
        participantKind: participant.kind,
        publicationCount: participant.trackPublications.size,
      });
      if (participant.kind === ParticipantKind.AGENT) {
        console.info('[captions] participant ignored: agent participant', {
          participantIdentity: participant.identity,
        });
        return;
      }

      let microphoneCount = 0;
      for (const publication of participant.trackPublications.values()) {
        if (publication.track && publication.source === TrackSource.SOURCE_MICROPHONE) {
          microphoneCount += 1;
          startTrack(publication.track, publication, participant);
        }
      }
      console.info('[captions] participant entrypoint complete', {
        participantIdentity: participant.identity,
        microphoneCount,
      });
    };

    ctx.room.on(RoomEvent.TrackSubscribed, startTrack);
    ctx.room.on(RoomEvent.TrackUnsubscribed, stopTrack);
    ctx.room.on(RoomEvent.ParticipantDisconnected, stopParticipant);
    ctx.addParticipantEntrypoint(trackParticipant);

    console.info('[captions] connecting with SUBSCRIBE_ALL');
    await ctx.connect(undefined, AutoSubscribe.SUBSCRIBE_ALL);
    console.info('[captions] connected', {
      roomName: ctx.room.name,
      remoteParticipantCount: ctx.room.remoteParticipants.size,
    });
  },
});

class TranscriptionWorker {
  private readonly stt = new inference.STT({
    model: 'assemblyai/universal-3-6-pro',
    language: 'en',
  });
  private readonly stream = this.stt.stream();
  private audioStream?: AudioStream;
  private stopped = false;
  private segmentNumber = 0;
  private readonly localParticipant: LocalParticipant;
  readonly participantIdentity: string;
  private readonly trackSid: string;
  private readonly track: RemoteTrack;

  constructor(
    localParticipant: LocalParticipant,
    participantIdentity: string,
    trackSid: string,
    track: RemoteTrack,
  ) {
    this.localParticipant = localParticipant;
    this.participantIdentity = participantIdentity;
    this.trackSid = trackSid;
    this.track = track;
  }

  async start() {
    console.info('[captions] creating audio stream', {
      participantIdentity: this.participantIdentity,
      trackSid: this.trackSid,
    });
    this.audioStream = new AudioStream(this.track, {
      sampleRate: 16_000,
      numChannels: 1,
      noiseCancellation: audioEnhancement({ model: EnhancerModel.QuailVfS }),
    });
    let frameCount = 0;
    let lastFrameLog = Date.now();

    void this.readTranscriptions().catch((error: unknown) => {
      console.error(
        `Transcription publishing failed for participant ${this.participantIdentity}:`,
        error,
      );
      this.stop();
    });
    for await (const frame of this.audioStream) {
      if (this.stopped) break;
      this.stream.pushFrame(frame);
      frameCount += 1;
      if (Date.now() - lastFrameLog >= 5000) {
        console.info('[captions] audio frames received', {
          participantIdentity: this.participantIdentity,
          trackSid: this.trackSid,
          frameCount,
        });
        lastFrameLog = Date.now();
      }
    }
    console.info('[captions] audio stream ended', {
      participantIdentity: this.participantIdentity,
      trackSid: this.trackSid,
      frameCount,
      stopped: this.stopped,
    });
    this.stream.endInput();
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    console.info('[captions] stopping worker', {
      participantIdentity: this.participantIdentity,
      trackSid: this.trackSid,
      segmentNumber: this.segmentNumber,
    });
    this.audioStream?.cancel();
    this.stream.close();
    void this.stt.close();
  }

  private async readTranscriptions() {
    for await (const event of this.stream) {
      const text = event.alternatives?.[0]?.text?.trim();
      console.info('[captions] STT event', {
        participantIdentity: this.participantIdentity,
        trackSid: this.trackSid,
        eventType: event.type,
        hasText: Boolean(text),
        textLength: text?.length ?? 0,
        requestId: event.requestId,
      });
      if (
        this.stopped ||
        !text ||
        (event.type !== INTERIM_TRANSCRIPT && event.type !== FINAL_TRANSCRIPT)
      ) {
        console.info('[captions] STT event ignored', {
          participantIdentity: this.participantIdentity,
          trackSid: this.trackSid,
          eventType: event.type,
          reason: this.stopped ? 'worker stopped' : !text ? 'empty text' : 'unsupported event type',
        });
        continue;
      }

      const final = event.type === FINAL_TRANSCRIPT;
      const segmentId =
        event.requestId ?? `${this.trackSid}-${this.segmentNumber}${final ? '-final' : ''}`;
      console.info('[captions] publishing transcription text stream', {
        participantIdentity: this.participantIdentity,
        trackSid: this.trackSid,
        segmentId,
        final,
        textLength: text.length,
      });
      const writer = await this.localParticipant.streamText({
        topic: 'lk.transcription',
        senderIdentity: this.participantIdentity,
        attributes: {
          'lk.transcribed_track_id': this.trackSid,
          'lk.transcription_final': String(final),
          'lk.segment_id': segmentId,
        },
      });
      await writer.write(text);
      await writer.close();

      if (final) {
        this.segmentNumber += 1;
      }
      console.info('[captions] transcription text stream published', {
        participantIdentity: this.participantIdentity,
        trackSid: this.trackSid,
        segmentId,
        final,
      });
    }
  }
}

cli.runApp(
  new ServerOptions({
    agent: fileURLToPath(import.meta.url),
    agentName: 'my-voice-agent',
  }),
);
