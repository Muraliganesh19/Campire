import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:shelf/shelf.dart';
import 'package:shelf/shelf_io.dart' as shelf_io;
import 'package:shelf_web_socket/shelf_web_socket.dart';
import 'package:shelf_multipart/shelf_multipart.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:crypto/crypto.dart';

import 'package:campire/models/sync_message.dart';
import 'package:campire/core/logger.dart';
import 'package:campire/network/network_constants.dart';

class HostServerService {
  HostServerService({
    required this.onMessageReceived,
    required this.onDeviceConnected,
    required this.onDeviceDisconnected,
  });

  final void Function(SyncMessage, WebSocketChannel) onMessageReceived;
  final void Function(String, String) onDeviceConnected;
  final void Function(String) onDeviceDisconnected;

  HttpServer? _server;
  final Map<String, WebSocketChannel> _activeChannels = {};
  int _unknownCamCount = 0;

  Future<void> start() async {
    final handler = const Pipeline()
        .addMiddleware(logRequests())
        .addHandler(_router);

    // Run on tcpCommandPort (45011)
    _server = await shelf_io.serve(
      handler,
      InternetAddress.anyIPv4,
      NetworkConstants.tcpCommandPort,
    );
    SyncLogger.info('HostServer listening on port ${_server!.port}');
  }

  Future<Response> _router(Request request) async {
    if (request.url.path == 'ws') {
      return webSocketHandler((WebSocketChannel webSocket, String? protocol) {
        String? deviceId;

        webSocket.stream.listen(
          (message) {
            try {
              final json =
                  jsonDecode(message as String) as Map<String, dynamic>;
              final syncMsg = SyncMessage.fromJson(json);

              if (deviceId == null) {
                deviceId = syncMsg.senderDeviceId;
                String deviceName = syncMsg.payload['deviceName'] as String? ?? '';
                if (deviceName.trim().isEmpty || deviceName == 'Unknown Camera') {
                  _unknownCamCount++;
                  deviceName = 'Unknown Cam $_unknownCamCount';
                }
                _activeChannels[deviceId!] = webSocket;
                onDeviceConnected(deviceId!, deviceName);
              }
              onMessageReceived(syncMsg, webSocket);
            } catch (e, stack) {
              SyncLogger.error('Error parsing WS message', e, stack);
            }
          },
          onDone: () {
            if (deviceId != null) {
              _activeChannels.remove(deviceId);
              onDeviceDisconnected(deviceId!);
            }
          },
          onError: (Object e, StackTrace stack) {
            SyncLogger.error('WebSocket channel error', e, stack);
          },
        );
      })(request);
    } else if (request.url.path == 'upload' && request.method == 'POST') {
      if (request.multipart() == null) {
        return Response.badRequest(body: 'Not a multipart request');
      }

      final expectedHash = request.headers['x-file-hash'];
      if (expectedHash == null) {
        return Response.badRequest(body: 'Missing x-file-hash header');
      }

      final tempPath = '${Directory.systemTemp.path}${Platform.pathSeparator}campire_uploads';
      final tempDir = Directory(tempPath);
      if (!await tempDir.exists()) await tempDir.create(recursive: true);

      File? savedFile;

      try {
        await for (final part in request.multipart()!.parts) {
          final contentDisposition = part.headers['content-disposition'];
          if (contentDisposition != null &&
              contentDisposition.contains('name="video"')) {
            // Extract filename rudimentarily
            String filename =
                'upload_${DateTime.now().millisecondsSinceEpoch}.mp4';
            final filenameMatch = RegExp(
              r'filename="([^"]+)"',
            ).firstMatch(contentDisposition);
            if (filenameMatch != null) {
              filename = filenameMatch.group(1)!;
            }

            savedFile = File('$tempPath${Platform.pathSeparator}$filename');
            final sink = savedFile.openWrite();
            await part.pipe(sink);
          }
        }
      } catch (e, stack) {
        SyncLogger.error('Upload error', e, stack);
        return Response.internalServerError(body: 'Upload failed');
      }

      if (savedFile == null) {
        return Response.badRequest(body: 'No video file found in payload');
      }

      // Verify SHA-256
      final bytes = await savedFile.readAsBytes();
      final hash = sha256.convert(bytes).toString();

      if (hash != expectedHash) {
        SyncLogger.error('Hash mismatch! Expected: $expectedHash, got: $hash');
        return Response.internalServerError(body: 'Hash mismatch');
      }

      SyncLogger.info(
        'Successfully uploaded and verified file at ${savedFile.path}',
      );
      print('=== VIDEO UPLOADED ===');
      print('Absolute path: ${savedFile.absolute.path}');
      print('======================');
      return Response.ok('Upload successful');
    }

    return Response.notFound('Not found');
  }

  Future<void> broadcast(SyncMessage message) async {
    final payload = jsonEncode(message.toJson());
    for (final channel in _activeChannels.values) {
      channel.sink.add(payload);
    }
  }

  Future<void> sendTo(String deviceId, SyncMessage message) async {
    final channel = _activeChannels[deviceId];
    if (channel != null) {
      channel.sink.add(jsonEncode(message.toJson()));
    }
  }

  void stop() {
    _server?.close(force: true);
    for (final channel in _activeChannels.values) {
      channel.sink.close();
    }
    _activeChannels.clear();
    _unknownCamCount = 0;
    SyncLogger.info('HostServer stopped');
  }
}
