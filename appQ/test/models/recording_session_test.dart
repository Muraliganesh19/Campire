import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:campire/models/recording_session.dart';

void main() {
  group('RecordingSession JSON round-trip', () {
    test('basic session round-trips', () {
      const session = RecordingSession(
        sessionId: 'sess-001',
        scheduledStartMicros: 1719849602000000,
        participatingDeviceIds: ['ctrl-001', 'remote-002', 'remote-003'],
      );
      final roundTripped = RecordingSession.fromJson(session.toJson());
      expect(roundTripped, equals(session));
    });

    test('single-device session round-trips', () {
      const session = RecordingSession(
        sessionId: 'sess-solo',
        scheduledStartMicros: 0,
        participatingDeviceIds: ['ctrl-001'],
      );
      final roundTripped = RecordingSession.fromJson(session.toJson());
      expect(roundTripped, equals(session));
    });

    test('round-trip survives JSON string encoding', () {
      const session = RecordingSession(
        sessionId: 'sess-002',
        scheduledStartMicros: 1719849999999999,
        participatingDeviceIds: ['a', 'b', 'c', 'd', 'e', 'f'],
      );
      final jsonString = jsonEncode(session.toJson());
      final decoded = RecordingSession.fromJson(
        jsonDecode(jsonString) as Map<String, dynamic>,
      );
      expect(decoded, equals(session));
    });

    test('device ID ordering is preserved', () {
      const ids = ['z-device', 'a-device', 'm-device'];
      const session = RecordingSession(
        sessionId: 'sess-order',
        scheduledStartMicros: 42,
        participatingDeviceIds: ids,
      );
      final roundTripped = RecordingSession.fromJson(session.toJson());
      expect(roundTripped.participatingDeviceIds, orderedEquals(ids));
    });
  });

  group('RecordingSession toJson', () {
    test('produces expected keys and types', () {
      const session = RecordingSession(
        sessionId: 'sess-001',
        scheduledStartMicros: 1719849602000000,
        participatingDeviceIds: ['ctrl-001'],
      );
      final json = session.toJson();
      expect(json['sessionId'], isA<String>());
      expect(json['scheduledStartMicros'], isA<int>());
      expect(json['participatingDeviceIds'], isA<List>());
    });
  });

  group('RecordingSession copyWith', () {
    const original = RecordingSession(
      sessionId: 'sess-001',
      scheduledStartMicros: 100,
      participatingDeviceIds: ['a'],
    );

    test('replaces only specified fields', () {
      final updated = original.copyWith(
        scheduledStartMicros: 200,
      );
      expect(updated.sessionId, original.sessionId);
      expect(updated.scheduledStartMicros, 200);
      expect(updated.participatingDeviceIds, original.participatingDeviceIds);
    });

    test('can replace device list', () {
      final updated = original.copyWith(
        participatingDeviceIds: ['a', 'b'],
      );
      expect(updated.participatingDeviceIds, ['a', 'b']);
    });
  });

  group('RecordingSession equality', () {
    test('identical values are equal', () {
      const a = RecordingSession(
        sessionId: 's', scheduledStartMicros: 1, participatingDeviceIds: ['x'],
      );
      const b = RecordingSession(
        sessionId: 's', scheduledStartMicros: 1, participatingDeviceIds: ['x'],
      );
      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test('different device lists are not equal', () {
      const a = RecordingSession(
        sessionId: 's', scheduledStartMicros: 1, participatingDeviceIds: ['x'],
      );
      const b = RecordingSession(
        sessionId: 's', scheduledStartMicros: 1, participatingDeviceIds: ['y'],
      );
      expect(a, isNot(equals(b)));
    });

    test('different session IDs are not equal', () {
      const a = RecordingSession(
        sessionId: 'a', scheduledStartMicros: 1, participatingDeviceIds: ['x'],
      );
      const b = RecordingSession(
        sessionId: 'b', scheduledStartMicros: 1, participatingDeviceIds: ['x'],
      );
      expect(a, isNot(equals(b)));
    });
  });
}
