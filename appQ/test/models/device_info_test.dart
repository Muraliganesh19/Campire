import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:campire/models/device_info.dart';

void main() {
  group('DeviceRole', () {
    test('fromString parses valid values', () {
      expect(DeviceRole.fromString('controller'), DeviceRole.controller);
      expect(DeviceRole.fromString('remote'), DeviceRole.remote);
    });

    test('fromString returns remote for unknown value', () {
      expect(DeviceRole.fromString('unknown'), DeviceRole.remote);
    });
  });

  group('ConnectionStatus', () {
    test('fromString parses all four values', () {
      expect(ConnectionStatus.fromString('disconnected'), ConnectionStatus.disconnected);
      expect(ConnectionStatus.fromString('connecting'), ConnectionStatus.connecting);
      expect(ConnectionStatus.fromString('connected'), ConnectionStatus.connected);
      expect(ConnectionStatus.fromString('synced'), ConnectionStatus.synced);
    });

    test('fromString returns disconnected for unknown value', () {
      expect(ConnectionStatus.fromString('bogus'), ConnectionStatus.disconnected);
    });
  });

  group('DeviceInfo JSON round-trip', () {
    const device = DeviceInfo(
      id: 'dev-001',
      displayName: 'Pixel 7',
      ipAddress: '192.168.1.42',
      role: DeviceRole.controller,
      connectionStatus: ConnectionStatus.synced,
    );

    test('toJson produces expected keys and values', () {
      final json = device.toJson();
      expect(json['id'], 'dev-001');
      expect(json['displayName'], 'Pixel 7');
      expect(json['ipAddress'], '192.168.1.42');
      expect(json['role'], 'controller');
      expect(json['connectionStatus'], 'synced');
    });

    test('fromJson(toJson) round-trip produces equal object', () {
      final roundTripped = DeviceInfo.fromJson(device.toJson());
      expect(roundTripped, equals(device));
    });

    test('round-trip survives JSON string encoding', () {
      final jsonString = jsonEncode(device.toJson());
      final decoded = DeviceInfo.fromJson(
        jsonDecode(jsonString) as Map<String, dynamic>,
      );
      expect(decoded, equals(device));
    });

    test('round-trip for remote / disconnected device', () {
      const remote = DeviceInfo(
        id: 'dev-002',
        displayName: 'Galaxy S23',
        ipAddress: '192.168.1.99',
        role: DeviceRole.remote,
        connectionStatus: ConnectionStatus.disconnected,
      );
      expect(DeviceInfo.fromJson(remote.toJson()), equals(remote));
    });
  });

  group('DeviceInfo copyWith', () {
    const original = DeviceInfo(
      id: 'dev-001',
      displayName: 'Pixel 7',
      ipAddress: '192.168.1.42',
      role: DeviceRole.remote,
      connectionStatus: ConnectionStatus.disconnected,
    );

    test('replaces only specified fields', () {
      final updated = original.copyWith(
        connectionStatus: ConnectionStatus.connected,
      );
      expect(updated.id, original.id);
      expect(updated.displayName, original.displayName);
      expect(updated.connectionStatus, ConnectionStatus.connected);
    });
  });

  group('DeviceInfo equality', () {
    test('identical values are equal', () {
      const a = DeviceInfo(
        id: 'x', displayName: 'X', ipAddress: '1.2.3.4',
        role: DeviceRole.remote, connectionStatus: ConnectionStatus.connecting,
      );
      const b = DeviceInfo(
        id: 'x', displayName: 'X', ipAddress: '1.2.3.4',
        role: DeviceRole.remote, connectionStatus: ConnectionStatus.connecting,
      );
      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test('different values are not equal', () {
      const a = DeviceInfo(
        id: 'x', displayName: 'X', ipAddress: '1.2.3.4',
        role: DeviceRole.remote, connectionStatus: ConnectionStatus.connecting,
      );
      const b = DeviceInfo(
        id: 'y', displayName: 'X', ipAddress: '1.2.3.4',
        role: DeviceRole.remote, connectionStatus: ConnectionStatus.connecting,
      );
      expect(a, isNot(equals(b)));
    });
  });
}
