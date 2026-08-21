import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:crypto/crypto.dart';

import 'package:campire/models/sync_message.dart';
import 'package:campire/core/logger.dart';
import 'package:campire/network/network_constants.dart';

class NodeClientService {
  NodeClientService({
    required this.localDeviceId,
    required this.deviceName,
    required this.hostIp,
    required this.onMessageReceived,
    required this.onConnected,
    required this.onDisconnected,
  });

  final String localDeviceId;
  final String deviceName;
  final String hostIp;
  final void Function(SyncMessage) onMessageReceived;
  final void Function() onConnected;
  final void Function() onDisconnected;

  WebSocketChannel? _channel;

  Future<void> connect() async {
    final wsUri = Uri.parse('ws://$hostIp:${NetworkConstants.tcpCommandPort}/ws');
    try {
      _channel = WebSocketChannel.connect(wsUri);
      await _channel!.ready;
      
      SyncLogger.info('Connected to Host WebSocket at $wsUri');
      onConnected();

      // Send initial handshake so Host knows our deviceId and deviceName
      send(SyncMessage(
        type: SyncMessageType.handshakeRequest,
        senderDeviceId: localDeviceId,
        timestampMicros: DateTime.now().microsecondsSinceEpoch,
        payload: {'deviceName': deviceName},
      ));

      _channel!.stream.listen(
        (message) {
          try {
            final json = jsonDecode(message as String) as Map<String, dynamic>;
            final syncMsg = SyncMessage.fromJson(json);
            onMessageReceived(syncMsg);
          } catch (e, stack) {
            SyncLogger.error('Error parsing WS message from Host', e, stack);
          }
        },
        onDone: () {
          SyncLogger.info('Host WS connection closed');
          onDisconnected();
        },
        onError: (Object e, StackTrace stack) {
          SyncLogger.error('Host WS connection error', e, stack);
          onDisconnected();
        },
      );
    } catch (e, stack) {
      SyncLogger.error('Failed to connect to Host WS', e, stack);
      onDisconnected();
    }
  }

  void send(SyncMessage message) {
    if (_channel != null) {
      _channel!.sink.add(jsonEncode(message.toJson()));
    }
  }

  Future<bool> uploadVideo(File videoFile) async {
    final uploadUri = Uri.parse('http://$hostIp:${NetworkConstants.tcpCommandPort}/upload');
    SyncLogger.info('Starting upload to $uploadUri');

    try {
      // 1. Compute SHA-256
      final bytes = await videoFile.readAsBytes();
      final hash = sha256.convert(bytes).toString();
      SyncLogger.info('Computed SHA-256: $hash for ${videoFile.path}');

      // 2. Upload via multipart
      final request = http.MultipartRequest('POST', uploadUri)
        ..headers['x-file-hash'] = hash
        ..files.add(await http.MultipartFile.fromPath('video', videoFile.path));

      final response = await request.send();
      if (response.statusCode == 200) {
        SyncLogger.info('Upload successful for ${videoFile.path}');
        return true;
      } else {
        final respBody = await response.stream.bytesToString();
        SyncLogger.error('Upload failed with status ${response.statusCode}: $respBody');
        return false;
      }
    } catch (e, stack) {
      SyncLogger.error('Exception during upload', e, stack);
      return false;
    }
  }

  void disconnect() {
    _channel?.sink.close();
    _channel = null;
  }
}
