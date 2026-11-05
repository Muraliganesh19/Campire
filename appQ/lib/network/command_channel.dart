import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:campire/core/logger.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/network_constants.dart';

/// A single TCP connection for sending and receiving [SyncMessage] commands.
///
/// Messages are JSON-encoded, **newline-delimited** — one JSON object per
/// line. This gives explicit message boundaries over TCP's byte stream.
///
/// Used by both sides of the protocol:
/// * **Controller** creates a channel via [CommandChannel.connect] for each
///   remote it wants to command.
/// * **Remote** receives a channel from [CommandServer] when the controller
///   connects.
///
/// This channel handles `start_at`, `ack`, and `stop` messages only — clock
/// sync stays on its own UDP path in `clock_sync_service.dart`.
class CommandChannel {
  Socket? _socket;

  final StreamController<SyncMessage> _messageController =
      StreamController<SyncMessage>.broadcast();

  /// Buffered partial line data from the TCP stream. TCP may deliver partial
  /// lines or multiple lines in a single read.
  String _lineBuffer = '';

  StreamSubscription<List<int>>? _socketSubscription;
  bool _closed = false;

  /// Invoked when the underlying socket disconnects unexpectedly.
  /// The consumer (e.g. a provider) can use this to update connection status.
  void Function()? onDisconnected;

  CommandChannel._();

  // ── Public getters ──────────────────────────────────────────────────────

  /// Stream of incoming [SyncMessage] objects from the other end.
  Stream<SyncMessage> get messages => _messageController.stream;

  /// Whether the underlying socket is still open.
  bool get isConnected => _socket != null && !_closed;

  /// The remote IP address, if connected.
  String? get remoteAddress {
    try {
      return _socket?.remoteAddress.address;
    } catch (_) {
      return null;
    }
  }

  // ── Factory: wrap an already-accepted socket (remote side) ─────────────

  /// Creates a [CommandChannel] from an already-connected [socket].
  ///
  /// Used by [CommandServer] when the controller connects to a remote.
  factory CommandChannel.fromSocket(Socket socket) {
    final channel = CommandChannel._();
    channel._attachSocket(socket);
    return channel;
  }

  // ── Factory: controller connects to a remote ───────────────────────────

  /// Opens a TCP connection to [ipAddress]:[port] and returns a ready
  /// [CommandChannel].
  ///
  /// Retries up to [maxRetries] times on failure with exponential backoff
  /// (500 ms × 2^attempt). If all retries are exhausted the underlying
  /// [SocketException] is rethrown so the caller can mark the device as
  /// disconnected.
  static Future<CommandChannel> connect({
    required String ipAddress,
    int? port,
    Duration timeout = const Duration(seconds: 5),
    int maxRetries = 3,
  }) async {
    port ??= NetworkConstants.tcpCommandPort;
    final channel = CommandChannel._();
    await channel._connectWithRetry(ipAddress, port, timeout, maxRetries);
    return channel;
  }

  Future<void> _connectWithRetry(
    String ipAddress,
    int port,
    Duration timeout,
    int maxRetries,
  ) async {
    for (var attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        SyncLogger.info(
          'CommandChannel: connecting to $ipAddress:$port '
          '(attempt ${attempt + 1}/${maxRetries + 1})',
        );
        final socket = await Socket.connect(
          ipAddress,
          port,
          timeout: timeout,
        );
        _attachSocket(socket);
        SyncLogger.info(
          'CommandChannel: connected to $ipAddress:$port',
        );
        return;
      } catch (e, stack) {
        SyncLogger.error(
          'CommandChannel: connect attempt ${attempt + 1} to '
          '$ipAddress:$port failed',
          e,
          stack,
        );
        if (attempt < maxRetries) {
          // Exponential backoff: 500ms, 1s, 2s.
          final backoff = Duration(milliseconds: 500 * (1 << attempt));
          await Future<void>.delayed(backoff);
        } else {
          // All retries exhausted — rethrow so the caller can react.
          rethrow;
        }
      }
    }
  }

  // ── Socket lifecycle ───────────────────────────────────────────────────

  void _attachSocket(Socket socket) {
    _socket = socket;
    _closed = false;
    _lineBuffer = '';

    // We listen to raw bytes and manually split on newlines so that
    // partial TCP segments are handled correctly.
    _socketSubscription = socket.listen(
      _onData,
      onError: _onError,
      onDone: _onDone,
      cancelOnError: false,
    );
  }

  void _onData(List<int> data) {
    _lineBuffer += utf8.decode(data, allowMalformed: true);

    // Process every complete line in the buffer.
    while (true) {
      final newlineIndex = _lineBuffer.indexOf('\n');
      if (newlineIndex == -1) break;

      final line = _lineBuffer.substring(0, newlineIndex).trim();
      _lineBuffer = _lineBuffer.substring(newlineIndex + 1);

      if (line.isEmpty) continue;

      try {
        final json = jsonDecode(line) as Map<String, dynamic>;
        final message = SyncMessage.fromJson(json);
        _messageController.add(message);
      } catch (e) {
        SyncLogger.error(
          'CommandChannel: failed to parse message: $line',
          e,
        );
      }
    }
  }

  void _onError(Object error, StackTrace stack) {
    SyncLogger.error('CommandChannel: socket error', error, stack);
    _handleDisconnect();
  }

  void _onDone() {
    SyncLogger.info('CommandChannel: connection closed by remote');
    _handleDisconnect();
  }

  void _handleDisconnect() {
    if (_closed) return;
    _closed = true;
    onDisconnected?.call();
  }

  // ── Send ───────────────────────────────────────────────────────────────

  /// Serialises [message] to JSON, appends a newline, and writes it to the
  /// socket.
  Future<void> send(SyncMessage message) async {
    if (_closed || _socket == null) {
      SyncLogger.error('CommandChannel: cannot send — connection closed');
      return;
    }
    try {
      final encoded = '${jsonEncode(message.toJson())}\n';
      _socket!.write(encoded);
      await _socket!.flush();
    } catch (e, stack) {
      SyncLogger.error('CommandChannel: send failed', e, stack);
      _handleDisconnect();
    }
  }

  // ── Close ──────────────────────────────────────────────────────────────

  /// Closes the channel and releases resources.
  Future<void> close() async {
    if (_closed && _socket == null) return;
    _closed = true;

    await _socketSubscription?.cancel();
    _socketSubscription = null;

    try {
      _socket?.destroy();
    } catch (_) {}
    _socket = null;

    // Don't close the broadcast StreamController — consumers may still have
    // listeners they haven't cancelled. It will be garbage-collected.
  }

  /// Closes the channel **and** the message stream. Call when the channel
  /// will never be reused (e.g. on app dispose / role switch).
  Future<void> dispose() async {
    await close();
    await _messageController.close();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Remote-side TCP server
// ═══════════════════════════════════════════════════════════════════════════

/// A TCP server that listens for incoming controller connections on the
/// command port and wraps each accepted socket in a [CommandChannel].
///
/// Used by the **remote** device. Typically only one connection is accepted
/// (from the single controller), but the server handles the general case.
class CommandServer {
  ServerSocket? _serverSocket;
  final StreamController<CommandChannel> _connectionController =
      StreamController<CommandChannel>.broadcast();

  /// Stream of newly accepted [CommandChannel] connections.
  Stream<CommandChannel> get connections => _connectionController.stream;

  /// Whether the server is currently listening.
  bool get isListening => _serverSocket != null;

  /// Binds to [InternetAddress.anyIPv4] on [NetworkConstants.tcpCommandPort]
  /// and begins accepting connections.
  Future<void> start() async {
    if (_serverSocket != null) {
      SyncLogger.info('CommandServer: already listening — ignoring start()');
      return;
    }

    _serverSocket = await ServerSocket.bind(
      InternetAddress.anyIPv4,
      NetworkConstants.tcpCommandPort,
    );

    SyncLogger.info(
      'CommandServer: listening on port ${NetworkConstants.tcpCommandPort}',
    );

    _serverSocket!.listen(
      (socket) {
        SyncLogger.info(
          'CommandServer: accepted connection from '
          '${socket.remoteAddress.address}:${socket.remotePort}',
        );
        final channel = CommandChannel.fromSocket(socket);
        _connectionController.add(channel);
      },
      onError: (Object error, StackTrace stack) {
        SyncLogger.error('CommandServer: listener error', error, stack);
      },
    );
  }

  /// Stops the server and closes all resources.
  Future<void> stop() async {
    await _serverSocket?.close();
    _serverSocket = null;
    SyncLogger.info('CommandServer: stopped');
  }

  /// Stops the server and closes the connection stream.
  Future<void> dispose() async {
    await stop();
    await _connectionController.close();
  }
}
