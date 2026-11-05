import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:campire/core/logger.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/network_constants.dart';

/// Handles UDP-based device discovery on the local network.
///
/// Two modes of operation, matching the two app roles:
///
/// **Controller** — calls [startBroadcasting] which periodically sends a
/// `discover` [SyncMessage] to the subnet broadcast address. Incoming
/// `discover_response` packets are parsed and emitted on [discoveredRemotes].
///
/// **Remote** — calls [startListening] which waits for `discover` packets.
/// On receipt it unicasts a `discover_response` back to the controller's
/// IP (taken from the UDP datagram's source address, not from the payload).
///
/// All packet parsing is wrapped in try/catch so that malformed or foreign
/// UDP traffic is silently ignored.
///
/// This service handles discovery **only** — no clock sync, no reconnect
/// logic, no command channel setup.
class DiscoveryService {
  // ── Constructor ────────────────────────────────────────────────────────

  /// Creates a [DiscoveryService].
  ///
  /// [localDeviceId] and [localDisplayName] identify *this* device in the
  /// discovery protocol. [localRole] determines whether the service will
  /// broadcast (controller) or listen-and-reply (remote).
  DiscoveryService({
    required this.localDeviceId,
    required this.localDisplayName,
    required this.localRole,
  });

  final String localDeviceId;
  final String localDisplayName;
  final DeviceRole localRole;

  // ── Internal state ─────────────────────────────────────────────────────

  RawDatagramSocket? _socket;
  Timer? _broadcastTimer;
  bool _running = false;

  /// Controller-side stream of newly discovered remotes.
  /// Each emission represents a single `discover_response` that was
  /// successfully parsed. Duplicates (same device ID seen again) are
  /// *not* filtered here — that's the responsibility of the state layer
  /// ([DeviceListProvider]).
  final StreamController<DeviceInfo> _discoveredController =
      StreamController<DeviceInfo>.broadcast();

  /// Stream of [DeviceInfo] for each remote that responds to a discovery
  /// broadcast. Only meaningful when this device is a **controller**.
  Stream<DeviceInfo> get discoveredRemotes => _discoveredController.stream;

  /// Whether the service is currently broadcasting or listening.
  bool get isRunning => _running;

  // ── Public API ─────────────────────────────────────────────────────────

  /// Starts the discovery service.
  ///
  /// * **Controller**: binds a UDP socket, starts periodic broadcasting,
  ///   and listens for `discover_response` replies.
  /// * **Remote**: binds a UDP socket on the discovery port and listens
  ///   for `discover` broadcasts, replying to each with a
  ///   `discover_response`.
  Future<void> start() async {
    if (_running) {
      SyncLogger.info('DiscoveryService already running — ignoring start()');
      return;
    }

    const port = NetworkConstants.udpDiscoveryPort;

    // Bind on `anyIPv4` so we receive broadcasts regardless of interface.
    // `reuseAddress: true` lets multiple processes (or hot-restart) bind
    // to the same port during development.
    _socket = await RawDatagramSocket.bind(
      InternetAddress.anyIPv4,
      port,
      reuseAddress: true,
    );
    _socket!.broadcastEnabled = true;
    _running = true;

    SyncLogger.info(
      'DiscoveryService started as ${localRole.name} on port $port',
    );

    // Listen for incoming datagrams.
    _socket!.listen(
      (event) {
        if (event == RawSocketEvent.read) {
          _handleIncomingDatagram();
        }
      },
      onError: (Object error, StackTrace stack) {
        SyncLogger.error('DiscoveryService socket error', error, stack);
      },
      onDone: () {
        SyncLogger.info('DiscoveryService socket closed');
        _running = false;
      },
    );

    // Controller-only: start periodic broadcasting.
    if (localRole == DeviceRole.controller) {
      await _sendBroadcast(); // send one immediately
      _broadcastTimer = Timer.periodic(
        NetworkConstants.broadcastInterval,
        (_) => _sendBroadcast(),
      );
    }
  }

  /// Stops the discovery service and releases resources.
  void stop() {
    _broadcastTimer?.cancel();
    _broadcastTimer = null;
    _socket?.close();
    _socket = null;
    _running = false;
    SyncLogger.info('DiscoveryService stopped');
  }

  /// Releases the stream controller. Call when the service will no longer
  /// be used (e.g. on app dispose).
  void dispose() {
    stop();
    _discoveredController.close();
  }

  // ── Controller: broadcasting ───────────────────────────────────────────

  /// Sends a single `discover` broadcast packet.
  Future<void> _sendBroadcast() async {
    // Capture to a local variable to avoid TOCTOU race: stop() can null
    // _socket while we're awaiting broadcastAddress() below.
    final socket = _socket;
    if (socket == null) return;

    final message = SyncMessage(
      type: SyncMessageType.discover,
      senderDeviceId: localDeviceId,
      timestampMicros: DateTime.now().microsecondsSinceEpoch,
      payload: {
        'displayName': localDisplayName,
      },
    );

    final bytes = utf8.encode(jsonEncode(message.toJson()));
    final broadcastAddr = await NetworkConstants.broadcastAddress();

    // Defense-in-depth: re-check after the async gap — stop() may have
    // run and closed the socket while we were awaiting broadcastAddress().
    if (_socket == null) return;

    final sent = socket.send(
      bytes,
      broadcastAddr,
      NetworkConstants.udpDiscoveryPort,
    );

    if (sent > 0) {
      SyncLogger.info(
        'Broadcast discover → ${broadcastAddr.address}:${NetworkConstants.udpDiscoveryPort} ($sent bytes)',
      );
    }
  }

  // ── Incoming packet handling ───────────────────────────────────────────

  void _handleIncomingDatagram() {
    final datagram = _socket?.receive();
    if (datagram == null) return;

    try {
      final jsonString = utf8.decode(datagram.data);
      final json = jsonDecode(jsonString) as Map<String, dynamic>;
      final message = SyncMessage.fromJson(json);

      // Ignore our own packets (they loop back on some network configs).
      if (message.senderDeviceId == localDeviceId) return;

      switch (localRole) {
        case DeviceRole.controller:
          _handleAsController(message, datagram.address);
          break;
        case DeviceRole.remote:
          _handleAsRemote(message, datagram.address);
          break;
      }
    } catch (e, stack) {
      // Malformed or foreign UDP traffic — ignore silently.
      SyncLogger.error(
        'DiscoveryService: ignoring unparseable datagram from '
        '${datagram.address.address}:${datagram.port}',
        e,
        stack,
      );
    }
  }

  // ── Controller: handling responses ─────────────────────────────────────

  void _handleAsController(SyncMessage message, InternetAddress senderIp) {
    if (message.type != SyncMessageType.discoverResponse) return;

    // Build DeviceInfo using the packet's source IP (not any IP in the
    // payload) — the source IP is the one we can actually reach.
    final remoteInfo = DeviceInfo(
      id: message.senderDeviceId,
      displayName:
          (message.payload['displayName'] as String?) ?? 'Unknown Device',
      ipAddress: senderIp.address,
      role: DeviceRole.remote,
      connectionStatus: ConnectionStatus.disconnected,
    );

    SyncLogger.info('Discovered remote: $remoteInfo');
    _discoveredController.add(remoteInfo);
  }

  // ── Remote: handling broadcasts & replying ─────────────────────────────

  void _handleAsRemote(SyncMessage message, InternetAddress senderIp) {
    if (message.type != SyncMessageType.discover) return;

    SyncLogger.info(
      'Received discover from controller ${message.senderDeviceId} '
      'at ${senderIp.address}',
    );

    _sendDiscoverResponse(senderIp);
  }

  /// Unicasts a `discover_response` back to the controller that sent the
  /// broadcast. We reply to the controller's IP directly (not to the
  /// broadcast address).
  Future<void> _sendDiscoverResponse(InternetAddress controllerIp) async {
    if (_socket == null) return;

    final localIp = await NetworkConstants.localAddress();

    final response = SyncMessage(
      type: SyncMessageType.discoverResponse,
      senderDeviceId: localDeviceId,
      timestampMicros: DateTime.now().microsecondsSinceEpoch,
      payload: {
        'displayName': localDisplayName,
        'ipAddress': localIp.address,
        'role': DeviceRole.remote.name,
      },
    );

    final bytes = utf8.encode(jsonEncode(response.toJson()));

    final sent = _socket!.send(
      bytes,
      controllerIp,
      NetworkConstants.udpDiscoveryPort,
    );

    SyncLogger.info(
      'Sent discover_response → ${controllerIp.address}:${NetworkConstants.udpDiscoveryPort} ($sent bytes)',
    );
  }

  // ── Test helpers ───────────────────────────────────────────────────────

  /// Constructs the [SyncMessage] that a controller broadcasts during
  /// discovery. Exposed for unit testing message shape without needing
  /// a live socket.
  static SyncMessage buildDiscoverMessage({
    required String deviceId,
    required String displayName,
    required int timestampMicros,
  }) {
    return SyncMessage(
      type: SyncMessageType.discover,
      senderDeviceId: deviceId,
      timestampMicros: timestampMicros,
      payload: {'displayName': displayName},
    );
  }

  /// Constructs the [SyncMessage] that a remote sends in response to a
  /// controller's discover broadcast. Exposed for unit testing.
  static SyncMessage buildDiscoverResponse({
    required String deviceId,
    required String displayName,
    required String ipAddress,
    required int timestampMicros,
  }) {
    return SyncMessage(
      type: SyncMessageType.discoverResponse,
      senderDeviceId: deviceId,
      timestampMicros: timestampMicros,
      payload: {
        'displayName': displayName,
        'ipAddress': ipAddress,
        'role': DeviceRole.remote.name,
      },
    );
  }

  /// Parses a [DeviceInfo] from a `discover_response` [SyncMessage],
  /// using [senderIp] as the device's IP address (ignoring any IP in
  /// the payload). Exposed for unit testing.
  static DeviceInfo parseDiscoverResponse(
    SyncMessage message,
    String senderIp,
  ) {
    if (message.type != SyncMessageType.discoverResponse) {
      throw ArgumentError(
        'Expected discover_response, got ${message.type.wireValue}',
      );
    }

    return DeviceInfo(
      id: message.senderDeviceId,
      displayName:
          (message.payload['displayName'] as String?) ?? 'Unknown Device',
      ipAddress: senderIp,
      role: DeviceRole.remote,
      connectionStatus: ConnectionStatus.disconnected,
    );
  }
}
