import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:campire/models/sync_message.dart';

void main() {
  group('SyncMessageType', () {
    test('all types have distinct wire values', () {
      final wireValues = SyncMessageType.values.map((e) => e.wireValue).toSet();
      expect(wireValues.length, SyncMessageType.values.length);
    });

    test('fromWire parses every valid wire value', () {
      for (final type in SyncMessageType.values) {
        expect(SyncMessageType.fromWire(type.wireValue), type);
      }
    });

    test('fromWire throws on unknown value', () {
      expect(
        () => SyncMessageType.fromWire('invalid'),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('SyncMessage JSON round-trip', () {
    test('discover message round-trips', () {
      const msg = SyncMessage(
        type: SyncMessageType.discover,
        senderDeviceId: 'ctrl-001',
        timestampMicros: 1719849600000000,
      );
      final roundTripped = SyncMessage.fromJson(msg.toJson());
      expect(roundTripped, equals(msg));
    });

    test('discover_response with payload round-trips', () {
      const msg = SyncMessage(
        type: SyncMessageType.discoverResponse,
        senderDeviceId: 'remote-001',
        timestampMicros: 1719849600100000,
        payload: {'displayName': 'Galaxy S24', 'ipAddress': '192.168.1.42'},
      );
      final roundTripped = SyncMessage.fromJson(msg.toJson());
      expect(roundTripped, equals(msg));
    });

    test('handshake_request with payload round-trips', () {
      const msg = SyncMessage(
        type: SyncMessageType.handshakeRequest,
        senderDeviceId: 'ctrl-001',
        timestampMicros: 1719849600000000,
        payload: {'t1': 1719849600000000},
      );
      final roundTripped = SyncMessage.fromJson(msg.toJson());
      expect(roundTripped, equals(msg));
    });

    test('handshake_response with payload round-trips', () {
      const msg = SyncMessage(
        type: SyncMessageType.handshakeResponse,
        senderDeviceId: 'remote-002',
        timestampMicros: 1719849600500000,
        payload: {'t1': 1719849600000000, 't2': 1719849600200000},
      );
      final roundTripped = SyncMessage.fromJson(msg.toJson());
      expect(roundTripped, equals(msg));
    });

    test('start_at message round-trips', () {
      const msg = SyncMessage(
        type: SyncMessageType.startAt,
        senderDeviceId: 'ctrl-001',
        timestampMicros: 1719849601000000,
        payload: {
          'sessionId': 'sess-abc',
          'targetStartMicros': 1719849602000000,
        },
      );
      final roundTripped = SyncMessage.fromJson(msg.toJson());
      expect(roundTripped, equals(msg));
    });

    test('ack message round-trips', () {
      const msg = SyncMessage(
        type: SyncMessageType.ack,
        senderDeviceId: 'remote-003',
        timestampMicros: 1719849601100000,
        payload: {'sessionId': 'sess-abc'},
      );
      final roundTripped = SyncMessage.fromJson(msg.toJson());
      expect(roundTripped, equals(msg));
    });

    test('round-trip survives JSON string encoding', () {
      const msg = SyncMessage(
        type: SyncMessageType.startAt,
        senderDeviceId: 'ctrl-001',
        timestampMicros: 1719849601000000,
        payload: {'targetStartMicros': 1719849602000000},
      );
      final jsonString = jsonEncode(msg.toJson());
      final decoded = SyncMessage.fromJson(
        jsonDecode(jsonString) as Map<String, dynamic>,
      );
      expect(decoded, equals(msg));
    });

    test('empty payload is preserved', () {
      const msg = SyncMessage(
        type: SyncMessageType.discover,
        senderDeviceId: 'ctrl-001',
        timestampMicros: 1719849600000000,
      );
      final json = msg.toJson();
      expect(json['payload'], isA<Map>());
      expect((json['payload'] as Map).isEmpty, isTrue);

      final roundTripped = SyncMessage.fromJson(json);
      expect(roundTripped.payload, isEmpty);
    });

    test('fromJson tolerates missing payload key', () {
      final json = {
        'type': 'discover',
        'senderDeviceId': 'ctrl-001',
        'timestampMicros': 1719849600000000,
      };
      final msg = SyncMessage.fromJson(json);
      expect(msg.payload, isEmpty);
    });
  });

  group('SyncMessage wire format', () {
    test('type field uses snake_case wire values', () {
      const msg = SyncMessage(
        type: SyncMessageType.handshakeRequest,
        senderDeviceId: 'x',
        timestampMicros: 0,
      );
      expect(msg.toJson()['type'], 'handshake_request');
    });
  });

  group('SyncMessage equality', () {
    test('identical values are equal', () {
      const a = SyncMessage(
        type: SyncMessageType.ack,
        senderDeviceId: 'dev-1',
        timestampMicros: 100,
        payload: {'k': 'v'},
      );
      const b = SyncMessage(
        type: SyncMessageType.ack,
        senderDeviceId: 'dev-1',
        timestampMicros: 100,
        payload: {'k': 'v'},
      );
      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test('different payload values are not equal', () {
      const a = SyncMessage(
        type: SyncMessageType.ack,
        senderDeviceId: 'dev-1',
        timestampMicros: 100,
        payload: {'k': 'v1'},
      );
      const b = SyncMessage(
        type: SyncMessageType.ack,
        senderDeviceId: 'dev-1',
        timestampMicros: 100,
        payload: {'k': 'v2'},
      );
      expect(a, isNot(equals(b)));
    });
  });
}
