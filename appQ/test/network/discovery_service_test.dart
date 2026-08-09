import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/discovery_service.dart';

// NOTE: End-to-end discovery (controller broadcasting, remote receiving,
// remote replying, controller seeing the DeviceInfo on its stream) requires
// two physical devices on the same WiFi / hotspot network. That flow cannot
// be unit-tested in isolation because RawDatagramSocket behaviour depends on
// OS networking, interface availability, and broadcast routing. Verify
// manually by running the app in controller mode on one device and remote
// mode on another, then checking the SyncLogger output for the expected
// discover → discover_response round-trip.

void main() {
  // ── buildDiscoverMessage ───────────────────────────────────────────────

  group('DiscoveryService.buildDiscoverMessage', () {
    test('produces a discover SyncMessage with correct fields', () {
      final msg = DiscoveryService.buildDiscoverMessage(
        deviceId: 'ctrl-001',
        displayName: 'Pixel 8 Pro',
        timestampMicros: 1719849600000000,
      );

      expect(msg.type, SyncMessageType.discover);
      expect(msg.senderDeviceId, 'ctrl-001');
      expect(msg.timestampMicros, 1719849600000000);
      expect(msg.payload['displayName'], 'Pixel 8 Pro');
    });

    test('message JSON round-trips cleanly', () {
      final msg = DiscoveryService.buildDiscoverMessage(
        deviceId: 'ctrl-001',
        displayName: 'Pixel 8 Pro',
        timestampMicros: 1719849600000000,
      );

      final jsonString = jsonEncode(msg.toJson());
      final decoded = SyncMessage.fromJson(
        jsonDecode(jsonString) as Map<String, dynamic>,
      );
      expect(decoded, equals(msg));
    });

    test('wire type is "discover"', () {
      final msg = DiscoveryService.buildDiscoverMessage(
        deviceId: 'x',
        displayName: 'X',
        timestampMicros: 0,
      );
      expect(msg.toJson()['type'], 'discover');
    });
  });

  // ── buildDiscoverResponse ──────────────────────────────────────────────

  group('DiscoveryService.buildDiscoverResponse', () {
    test('produces a discover_response SyncMessage with correct fields', () {
      final msg = DiscoveryService.buildDiscoverResponse(
        deviceId: 'remote-002',
        displayName: 'Galaxy S24',
        ipAddress: '192.168.1.42',
        timestampMicros: 1719849600500000,
      );

      expect(msg.type, SyncMessageType.discoverResponse);
      expect(msg.senderDeviceId, 'remote-002');
      expect(msg.timestampMicros, 1719849600500000);
      expect(msg.payload['displayName'], 'Galaxy S24');
      expect(msg.payload['ipAddress'], '192.168.1.42');
      expect(msg.payload['role'], 'remote');
    });

    test('message JSON round-trips cleanly', () {
      final msg = DiscoveryService.buildDiscoverResponse(
        deviceId: 'remote-002',
        displayName: 'Galaxy S24',
        ipAddress: '192.168.1.42',
        timestampMicros: 1719849600500000,
      );

      final jsonString = jsonEncode(msg.toJson());
      final decoded = SyncMessage.fromJson(
        jsonDecode(jsonString) as Map<String, dynamic>,
      );
      expect(decoded, equals(msg));
    });

    test('wire type is "discover_response"', () {
      final msg = DiscoveryService.buildDiscoverResponse(
        deviceId: 'x',
        displayName: 'X',
        ipAddress: '0.0.0.0',
        timestampMicros: 0,
      );
      expect(msg.toJson()['type'], 'discover_response');
    });
  });

  // ── parseDiscoverResponse ──────────────────────────────────────────────

  group('DiscoveryService.parseDiscoverResponse', () {
    test('extracts DeviceInfo with sender IP (not payload IP)', () {
      // The payload carries a self-reported IP, but the parser should use
      // the senderIp argument (from the UDP datagram source address).
      final msg = DiscoveryService.buildDiscoverResponse(
        deviceId: 'remote-002',
        displayName: 'Galaxy S24',
        ipAddress: '10.0.0.99', // self-reported IP in payload
        timestampMicros: 1719849600500000,
      );

      final info = DiscoveryService.parseDiscoverResponse(
        msg,
        '192.168.1.42', // actual packet source IP
      );

      expect(info.id, 'remote-002');
      expect(info.displayName, 'Galaxy S24');
      expect(info.ipAddress, '192.168.1.42'); // should use sender IP
      expect(info.role, DeviceRole.remote);
      expect(info.connectionStatus, ConnectionStatus.disconnected);
    });

    test('uses "Unknown Device" when displayName is missing from payload', () {
      const msg = SyncMessage(
        type: SyncMessageType.discoverResponse,
        senderDeviceId: 'remote-003',
        timestampMicros: 0,
        payload: {}, // no displayName
      );

      final info = DiscoveryService.parseDiscoverResponse(msg, '1.2.3.4');
      expect(info.displayName, 'Unknown Device');
    });

    test('throws on non-discover_response message type', () {
      const msg = SyncMessage(
        type: SyncMessageType.discover,
        senderDeviceId: 'ctrl-001',
        timestampMicros: 0,
      );

      expect(
        () => DiscoveryService.parseDiscoverResponse(msg, '1.2.3.4'),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('parsed DeviceInfo survives its own JSON round-trip', () {
      final msg = DiscoveryService.buildDiscoverResponse(
        deviceId: 'remote-004',
        displayName: 'OnePlus 12',
        ipAddress: '10.0.0.5',
        timestampMicros: 42,
      );

      final info = DiscoveryService.parseDiscoverResponse(msg, '192.168.1.7');
      final roundTripped = DeviceInfo.fromJson(info.toJson());
      expect(roundTripped, equals(info));
    });
  });

  // ── Full discover → discover_response pipeline (message layer only) ───

  group('Discovery message pipeline', () {
    test('controller broadcast → remote parse → remote reply → controller parse', () {
      // 1. Controller builds a discover broadcast.
      final discover = DiscoveryService.buildDiscoverMessage(
        deviceId: 'ctrl-001',
        displayName: 'Controller Phone',
        timestampMicros: 1000000,
      );

      // 2. Simulate: remote receives it, verifies type.
      expect(discover.type, SyncMessageType.discover);
      expect(discover.senderDeviceId, 'ctrl-001');

      // 3. Remote builds a response.
      final response = DiscoveryService.buildDiscoverResponse(
        deviceId: 'remote-005',
        displayName: 'Remote Phone',
        ipAddress: '192.168.1.50',
        timestampMicros: 1000500,
      );

      // 4. Simulate: serialize over the wire and deserialize on arrival.
      final wire = jsonEncode(response.toJson());
      final received = SyncMessage.fromJson(
        jsonDecode(wire) as Map<String, dynamic>,
      );
      expect(received, equals(response));

      // 5. Controller parses the response into a DeviceInfo.
      final deviceInfo = DiscoveryService.parseDiscoverResponse(
        received,
        '192.168.1.50',
      );

      expect(deviceInfo.id, 'remote-005');
      expect(deviceInfo.displayName, 'Remote Phone');
      expect(deviceInfo.ipAddress, '192.168.1.50');
      expect(deviceInfo.role, DeviceRole.remote);
      expect(deviceInfo.connectionStatus, ConnectionStatus.disconnected);
    });
  });
}
