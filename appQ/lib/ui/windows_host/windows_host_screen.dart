import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'package:file_picker/file_picker.dart';

import 'package:campire/core/host_paths.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/models/sync_message.dart';
import 'package:campire/network/host_server_service.dart';
import 'package:campire/network/discovery_service.dart';
import 'package:campire/network/clock_sync_service.dart';
import 'package:campire/camera/recording_manager.dart';
import 'package:campire/services/mock_pipeline_service.dart';
import 'package:campire/state/device_list_provider.dart';

class WindowsHostScreen extends StatefulWidget {
  const WindowsHostScreen({super.key});

  @override
  State<WindowsHostScreen> createState() => _WindowsHostScreenState();
}

class _WindowsHostScreenState extends State<WindowsHostScreen> {
  // Mock I/O Constants
  static final String inputBasePath = HostPaths.inputDir;
  static final String outputBasePath = HostPaths.outputDir;
  int liveSessionCounter = 1;
  int fileSessionCounter = 1;
  String? latestSessionPrefix;
  bool isUploadReady = false;

  // Services
  late HostServerService _hostServerService;
  late DiscoveryService _discoveryService;
  late ClockSyncService _clockSyncService;
  late RecordingManager _recordingManager;
  final MockPipelineService _pipelineService = MockPipelineService();

  // State
  bool _isDarkMode = true;
  int? _fullScreenCameraIndex;

  bool _isRecording = false;
  bool _isAnalyzing = false;

  String _ballDecision = 'NIL';
  String _bowlerDecision = 'NIL';
  String _wicketDecision = 'NIL';

  // MediaKit
  final List<Player> _players = [];
  final List<VideoController> _videoControllers = [];

  @override
  void initState() {
    super.initState();
    // Initialize MediaKit players
    for (int i = 0; i < 4; i++) {
      final player = Player();
      _players.add(player);
      _videoControllers.add(VideoController(player));
    }
    _initServices();
  }

  void _initServices() {
    try {
      Directory(inputBasePath).createSync(recursive: true);
      Directory(outputBasePath).createSync(recursive: true);
    } catch (e) {
      debugPrint('Failed to create mock directories: $e');
    }

    _recordingManager = RecordingManager(localDeviceId: 'host');
    _clockSyncService = ClockSyncService(localDeviceId: 'host');
    _discoveryService = DiscoveryService(
      localDeviceId: 'host',
      localDisplayName: 'Windows Host',
      localRole: DeviceRole.controller,
    );

    _hostServerService = HostServerService(
      onMessageReceived: _handleMessage,
      onDeviceConnected: _handleDeviceConnected,
      onDeviceDisconnected: _handleDeviceDisconnected,
    );

    // Listen to UDP discovery
    _discoveryService.discoveredRemotes.listen((device) {
      if (!mounted) return;
      final provider = Provider.of<DeviceListProvider>(context, listen: false);
      if (provider.getDevice(device.id) == null) {
        provider.updateDevice(device);
      }
    });
  }

  void _resetState() {
    _hostServerService.stop();
    _discoveryService.dispose();
    _clockSyncService.dispose();
    Provider.of<DeviceListProvider>(context, listen: false).clear();

    _initServices();

    setState(() {
      _isRecording = false;
      _isAnalyzing = false;
      _ballDecision = 'NIL';
      _bowlerDecision = 'NIL';
      _wicketDecision = 'NIL';
    });
  }

  void _handleMessage(SyncMessage message, channel) {
    if (message.type == SyncMessageType.clockSyncResponse) {
      _clockSyncService.handleResponse(message);
    } else if (message.type == SyncMessageType.ack) {
      _recordingManager.handleAck(message);
    }
  }

  void _handleDeviceConnected(String deviceId, String deviceName) async {
    if (!mounted) return;
    final provider = Provider.of<DeviceListProvider>(context, listen: false);

    var device = provider.getDevice(deviceId);
    if (device == null) {
      device = DeviceInfo(
        id: deviceId,
        displayName: deviceName,
        ipAddress: 'Unknown',
        role: DeviceRole.remote,
        connectionStatus: ConnectionStatus.connected,
      );
    } else {
      device = device.copyWith(
        displayName: deviceName,
        connectionStatus: ConnectionStatus.connected,
      );
    }
    provider.updateDevice(device);

    // Auto-sync clock
    try {
      await _clockSyncService.computeOffset(device, (msg) async {
        await _hostServerService.sendTo(deviceId, msg);
      });
      provider.updateStatus(deviceId, ConnectionStatus.synced);
    } catch (e) {
      provider.updateStatus(deviceId, ConnectionStatus.disconnected);
    }
  }

  void _handleDeviceDisconnected(String deviceId) {
    if (!mounted) return;
    final provider = Provider.of<DeviceListProvider>(context, listen: false);
    provider.updateStatus(deviceId, ConnectionStatus.disconnected);
  }

  @override
  void dispose() {
    _hostServerService.stop();
    _discoveryService.dispose();
    _clockSyncService.dispose();
    for (var player in _players) {
      player.dispose();
    }
    super.dispose();
  }

  void _scan() {
    _hostServerService.start();
    _discoveryService.start();
  }

  void _toggleRecording() async {
    final provider = Provider.of<DeviceListProvider>(context, listen: false);
    final synced = provider.devices
        .where((d) => d.connectionStatus == ConnectionStatus.synced)
        .map((d) => d.id)
        .toList();

    if (_isRecording) {
      await _recordingManager.triggerStop(
        hostServerService: _hostServerService,
      );

      setState(() {
        _isRecording = false;
        _ballDecision = 'NIL';
        _bowlerDecision = 'NIL';
        _wicketDecision = 'NIL';
      });

      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) setState(() => isUploadReady = true);
      });
    } else {
      if (synced.isEmpty) return;
      await _recordingManager.triggerSyncRecording(
        connectedDeviceIds: synced,
        hostServerService: _hostServerService,
      );
      setState(() {
        _isRecording = true;
        isUploadReady = false;
      });
    }
  }

  void _clearDirectoryFiles(String path) {
    final dir = Directory(path);
    if (dir.existsSync()) {
      for (var entity in dir.listSync()) {
        if (entity is File) {
          try {
            entity.deleteSync();
          } catch (e) {
            print('Failed to delete file: $e');
          }
        }
      }
    }
  }

  Future<void> _uploadVideos() async {
    setState(() => isUploadReady = false);

    final tempPath = '${Directory.systemTemp.path}${Platform.pathSeparator}campire_uploads';
    final tempDir = Directory(tempPath);
    List<File> mp4Files = [];

    if (tempDir.existsSync()) {
      final files = tempDir.listSync();
      for (var entity in files) {
        if (entity is File && entity.path.toLowerCase().endsWith('.mp4')) {
          mp4Files.add(entity);
        }
      }
    }

    mp4Files.sort((a, b) => a.lastModifiedSync().compareTo(b.lastModifiedSync()));

    final prefix = 'liveip$liveSessionCounter';

    // Housekeeping: Clear I/O directories before new upload
    _clearDirectoryFiles(inputBasePath);
    _clearDirectoryFiles(outputBasePath);

    if (mp4Files.isEmpty) {
      print('No real uploads found in temp dir. Checking for fallback video...');
      File? fallbackFile;
      final inputDir = Directory(inputBasePath);
      if (inputDir.existsSync()) {
        final existingFiles = inputDir.listSync();
        for (var entity in existingFiles) {
          if (entity is File && entity.path.toLowerCase().endsWith('.mp4')) {
            fallbackFile = entity;
            break;
          }
        }
      }
      
      if (fallbackFile != null) {
         for (int i = 1; i <= 4; i++) {
            fallbackFile.copySync('$inputBasePath${Platform.pathSeparator}${prefix}_cam$i.mp4');
         }
      } else {
         print('WARNING: No .mp4 files found to upload and no fallback available.');
      }
    } else {
      for (int i = 0; i < 4 && i < mp4Files.length; i++) {
        final outputPath = '$inputBasePath${Platform.pathSeparator}${prefix}_cam${i + 1}.mp4';
        mp4Files[i].copySync(outputPath);
      }
      
      for (var f in mp4Files) {
        try {
          f.deleteSync();
        } catch (_) {}
      }
    }

    final jsonPath = '$inputBasePath${Platform.pathSeparator}$prefix.json';
    final jsonData = {
      "session_id": prefix,
      "bowler_stance": "NIL",
      "delivery": "NIL",
      "wicket": "NIL"
    };
    File(jsonPath).writeAsStringSync(jsonEncode(jsonData));

    setState(() {
      latestSessionPrefix = prefix;
      liveSessionCounter++;
    });
  }

  void _uploadFiles() async {
    String? selectedDirectory = await FilePicker.getDirectoryPath();
    if (selectedDirectory != null) {
      print('User selected directory: $selectedDirectory');
      
      if (selectedDirectory.toLowerCase() == inputBasePath.toLowerCase() ||
          selectedDirectory.toLowerCase() == outputBasePath.toLowerCase()) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Cannot upload directly from the Input or Output directories.'),
              backgroundColor: Colors.red,
            ),
          );
        }
        return;
      }
      
      final dir = Directory(selectedDirectory);
      final files = dir.listSync();

      List<File> videos = [];
      File? jsonFile;

      for (var entity in files) {
        if (entity is File) {
          final ext = entity.path.toLowerCase();
          if (ext.endsWith('.mp4') || ext.endsWith('.avi') || ext.endsWith('.mov')) {
            videos.add(entity);
          } else if (ext.endsWith('.json')) {
            jsonFile = entity;
          }
        }
      }

      videos.sort((a, b) => a.path.compareTo(b.path));

      final prefix = 'fileip$fileSessionCounter';

      // Housekeeping: Clear I/O directories before new upload
      _clearDirectoryFiles(inputBasePath);
      _clearDirectoryFiles(outputBasePath);

      for (int i = 0; i < 4 && i < videos.length; i++) {
        final originalVideo = videos[i];
        final inputVideoPath = '$inputBasePath${Platform.pathSeparator}${prefix}_cam${i + 1}.mp4';
        originalVideo.copySync(inputVideoPath);
        
        final originalPath = originalVideo.path;
        final lastDotIndex = originalPath.lastIndexOf('.');
        if (lastDotIndex != -1) {
          final xmlPath = '${originalPath.substring(0, lastDotIndex)}.xml';
          final xmlFile = File(xmlPath);
          if (xmlFile.existsSync()) {
            final inputXmlPath = '$inputBasePath${Platform.pathSeparator}${prefix}_cam${i + 1}.xml';
            xmlFile.copySync(inputXmlPath);
          }
        }
      }

      if (jsonFile != null) {
        final inputJsonPath = '$inputBasePath${Platform.pathSeparator}$prefix.json';
        jsonFile.copySync(inputJsonPath);
      } else {
        final jsonPath = '$inputBasePath${Platform.pathSeparator}$prefix.json';
        final jsonData = {
          "session_id": prefix,
          "bowler_stance": "NIL",
          "delivery": "NIL",
          "wicket": "NIL"
        };
        File(jsonPath).writeAsStringSync(jsonEncode(jsonData));
      }

      setState(() {
        latestSessionPrefix = prefix;
        fileSessionCounter++;
        _bowlerDecision = 'NIL';
        _ballDecision = 'NIL';
        _wicketDecision = 'NIL';
      });
    } else {
      print('User canceled directory selection.');
    }
  }

  void _analyze() async {
    if (latestSessionPrefix == null) {
       print('No session to analyze');
       return;
    }
    
    setState(() => _isAnalyzing = true);

    try {
      if (latestSessionPrefix!.startsWith('liveip')) {
        final opPrefix = latestSessionPrefix!.replaceFirst('liveip', 'liveop');
        
        for (int i = 1; i <= 4; i++) {
          final inPath = '$inputBasePath${Platform.pathSeparator}${latestSessionPrefix}_cam$i.mp4';
          final outPath = '$outputBasePath${Platform.pathSeparator}${opPrefix}_cam$i.mp4';
          final inFile = File(inPath);
          if (inFile.existsSync()) {
            inFile.copySync(outPath);
          }
        }
        
        final inJsonPath = '$inputBasePath${Platform.pathSeparator}$latestSessionPrefix.json';
        final outJsonPath = '$outputBasePath${Platform.pathSeparator}$opPrefix.json';
        final inJsonFile = File(inJsonPath);
        if (inJsonFile.existsSync()) {
          inJsonFile.copySync(outJsonPath);
          final content = File(outJsonPath).readAsStringSync();
          final Map<String, dynamic> data = jsonDecode(content);
          setState(() {
            _bowlerDecision = data['bowler_stance']?.toString() ?? 'NIL';
            _ballDecision = data['delivery']?.toString() ?? 'NIL';
            _wicketDecision = data['wicket']?.toString() ?? 'NIL';
          });
        }
        
        for (int i = 0; i < 4; i++) {
          final outPath = '$outputBasePath${Platform.pathSeparator}${opPrefix}_cam${i + 1}.mp4';
          if (File(outPath).existsSync()) {
            _players[i].open(Media(outPath));
            _players[i].play();
          }
        }
      } else if (latestSessionPrefix!.startsWith('fileip')) {
        print('Running videomaker.py...');
        try {
          final result = await Process.run(HostPaths.pythonCommand, [HostPaths.videomakerScript]);
          print('Python script output: ${result.stdout}');
          
          if (result.exitCode != 0) {
            final errorText = result.stderr.toString().isNotEmpty ? result.stderr.toString() : 'Unknown Python error. Exit code: ${result.exitCode}';
            print('Python script error: $errorText');
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('Python Script Failed:\n$errorText'),
                  backgroundColor: Colors.red,
                  duration: const Duration(seconds: 10),
                ),
              );
            }
            setState(() => _isAnalyzing = false);
            return;
          }
        } catch (e) {
          print('Exception running python script: $e');
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('Failed to execute python script:\n$e'),
                backgroundColor: Colors.red,
                duration: const Duration(seconds: 10),
              ),
            );
          }
          setState(() => _isAnalyzing = false);
          return;
        }
        
        // External Python script saves output using the original input prefix
        final outJsonPath = '$outputBasePath${Platform.pathSeparator}$latestSessionPrefix.json';
        final inJsonPath = '$inputBasePath${Platform.pathSeparator}$latestSessionPrefix.json';
        bool allFilesExist = true;
        
        File jsonToRead = File(outJsonPath);
        if (!jsonToRead.existsSync() && File(inJsonPath).existsSync()) {
          // Fallback: If python script didn't create an output JSON, use the one from Input
          jsonToRead = File(inJsonPath);
        }
        
        if (jsonToRead.existsSync()) {
          final content = jsonToRead.readAsStringSync();
          final Map<String, dynamic> data = jsonDecode(content);
          setState(() {
            _bowlerDecision = data['bowler_stance']?.toString() ?? 'NIL';
            _ballDecision = data['delivery']?.toString() ?? 'NIL';
            _wicketDecision = data['wicket']?.toString() ?? 'NIL';
          });
        } else {
          print('Warning: JSON not found at $outJsonPath or $inJsonPath');
          setState(() {
            _bowlerDecision = 'NIL';
            _ballDecision = 'NIL';
            _wicketDecision = 'NIL';
          });
        }
        
        for (int i = 0; i < 4; i++) {
          final outPath = '$outputBasePath${Platform.pathSeparator}${latestSessionPrefix}_cam${i + 1}.mp4';
          if (!File(outPath).existsSync()) {
            allFilesExist = false;
          }
        }
        
        if (!allFilesExist) {
          if (mounted) {
             ScaffoldMessenger.of(context).showSnackBar(
               const SnackBar(
                 content: Text('Output files not found. The Python script may have failed silently or output wrong filenames.'),
                 backgroundColor: Colors.orange,
                 duration: Duration(seconds: 8),
               ),
             );
          }
          setState(() => _isAnalyzing = false);
          return;
        }
        
        for (int i = 0; i < 4; i++) {
          final outPath = '$outputBasePath${Platform.pathSeparator}${latestSessionPrefix}_cam${i + 1}.mp4';
          _players[i].open(Media(outPath));
          _players[i].play();
        }
      }
    } catch (e) {
      print('Error during analysis: $e');
    }

    setState(() {
      _isAnalyzing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final bgColor = _isDarkMode
        ? const Color(0xFF000000)
        : const Color(0xFFF0F0F0);
    final cardColor = _isDarkMode
        ? const Color(0xFF1C1C1E)
        : const Color(0xFFE0E0E0);
    final textColor = _isDarkMode ? Colors.white : Colors.black;

    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: Stack(
          children: [
            Column(
              children: [
                _buildHeader(cardColor, textColor),
                Expanded(
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _buildLeftPanel(cardColor, textColor),
                      Expanded(
                        child: Column(
                          children: [
                            Expanded(
                              child: _buildVideoGrid(cardColor, textColor),
                            ),
                            _buildDecisionTiles(cardColor, textColor),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            if (_fullScreenCameraIndex != null) _buildFullScreenOverlay(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(Color cardColor, Color textColor) {
    return Container(
      margin: const EdgeInsets.all(16.0),
      padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16.0),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(40),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: Text(
                  'HOME',
                  style: TextStyle(
                    color: textColor,
                    decoration: TextDecoration.underline,
                  ),
                ),
              ),
              IconButton(
                icon: Icon(Icons.refresh, color: textColor),
                onPressed: _resetState,
              ),
            ],
          ),
          Text(
            'CAMPIRE',
            style: TextStyle(
              color: textColor,
              fontSize: 32,
              fontWeight: FontWeight.w900,
              letterSpacing: 2.0,
            ),
          ),
          Row(
            children: [
              IconButton(
                icon: Icon(
                  _isDarkMode ? Icons.nightlight_round : Icons.wb_sunny,
                  color: textColor,
                ),
                onPressed: () => setState(() => _isDarkMode = !_isDarkMode),
              ),
              const SizedBox(width: 16),
              Text(
                'V 1.0.1',
                style: TextStyle(
                  color: textColor,
                  decoration: TextDecoration.underline,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildLeftPanel(Color cardColor, Color textColor) {
    return Container(
      width: 320,
      margin: const EdgeInsets.only(left: 16.0, bottom: 16.0, right: 16.0),
      child: CustomScrollView(
        slivers: [
          SliverFillRemaining(
            hasScrollBody: false,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Remote Camera List
          Container(
            padding: const EdgeInsets.all(16.0),
            decoration: BoxDecoration(
              color: cardColor,
              borderRadius: BorderRadius.circular(24),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Text(
                    'REMOTE CAMERA',
                    style: TextStyle(
                      color: textColor,
                      fontWeight: FontWeight.bold,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                ...List.generate(
                  4,
                  (index) => _buildCameraSlot(index, cardColor, textColor),
                ),
                const SizedBox(height: 8),
                Center(
                  child: ElevatedButton(
                    onPressed: _scan,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _isDarkMode
                          ? const Color(0xFF2C2C2E)
                          : Colors.grey[300],
                      foregroundColor: textColor,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                      padding: const EdgeInsets.symmetric(horizontal: 32),
                    ),
                    child: const Text('SCAN'),
                  ),
                ),
              ],
            ),
          ),
          const Spacer(), // Pushes action buttons to the bottom
          // Actions
          Container(
            padding: const EdgeInsets.all(16.0),
            decoration: BoxDecoration(
              color: cardColor,
              borderRadius: BorderRadius.circular(24),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                ElevatedButton(
                  onPressed: _toggleRecording,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isDarkMode
                        ? const Color(0xFF2C2C2E)
                        : Colors.grey[300],
                    foregroundColor: textColor,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  child: RichText(
                    text: TextSpan(
                      style: TextStyle(
                        color: textColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                      children: [
                        TextSpan(text: _isRecording ? 'STOP ' : 'START '),
                        TextSpan(
                          text: 'REC',
                          style: TextStyle(
                            color: _isRecording ? Colors.red : Colors.redAccent,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: _uploadVideos,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: isUploadReady 
                        ? Colors.green 
                        : (_isDarkMode ? const Color(0xFF2C2C2E) : Colors.grey[300]),
                    foregroundColor: isUploadReady ? Colors.white : textColor,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  child: const Text(
                    'UPLOAD VIDEOS',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: _uploadFiles,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isDarkMode
                        ? const Color(0xFF2C2C2E)
                        : Colors.grey[300],
                    foregroundColor: textColor,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  child: const Text(
                    'UPLOAD FILES',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: _isAnalyzing ? null : _analyze,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isDarkMode
                        ? const Color(0xFF2C2C2E)
                        : Colors.grey[300],
                    foregroundColor: textColor,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  child: _isAnalyzing
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text(
                          'ANALYZE',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                ),
              ],
            ),
          ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraSlot(int index, Color cardColor, Color textColor) {
    return Consumer<DeviceListProvider>(
      builder: (context, provider, _) {
        final devices = provider.devices;
        DeviceInfo? device;
        if (index < devices.length) {
          device = devices[index];
        }

        final isConnected =
            device?.connectionStatus == ConnectionStatus.connected ||
            device?.connectionStatus == ConnectionStatus.synced;

        String statusText = '---';
        Color statusDotColor = Colors.red;
        Color statusPillBg = _isDarkMode
            ? const Color(0xFF2C2C2E)
            : Colors.grey[400]!;
        Color statusPillText = Colors.red;

        if (isConnected) {
          statusDotColor = Colors.green;
          if (_isRecording) {
            statusText = 'REC';
            statusPillText = Colors.red;
          } else {
            statusText = 'STD';
            statusPillText = Colors.deepPurpleAccent;
          }
        }

        final deviceName = device?.displayName ?? '---';
        final displayName = deviceName.length > 8
            ? '${deviceName.substring(0, 6)}...'
            : deviceName;

        return Padding(
          padding: const EdgeInsets.only(bottom: 12.0),
          child: Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: _isDarkMode
                      ? const Color(0xFF2C2C2E)
                      : Colors.grey[400],
                  shape: BoxShape.circle,
                ),
                alignment: Alignment.center,
                child: Text(
                  '${index + 1}',
                  style: TextStyle(color: textColor, fontSize: 12),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    color: _isDarkMode
                        ? const Color(0xFF2C2C2E)
                        : Colors.grey[400],
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          displayName,
                          style: TextStyle(color: textColor, fontSize: 14),
                        ),
                      ),
                      const Icon(
                        Icons.videocam_outlined,
                        color: Colors.grey,
                        size: 16,
                      ),
                      const SizedBox(width: 8),
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          color: statusDotColor,
                          shape: BoxShape.circle,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Container(
                width: 50,
                padding: const EdgeInsets.symmetric(vertical: 8),
                decoration: BoxDecoration(
                  color: statusPillBg,
                  borderRadius: BorderRadius.circular(16),
                ),
                alignment: Alignment.center,
                child: Text(
                  statusText,
                  style: TextStyle(
                    color: statusPillText,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildVideoGrid(Color cardColor, Color textColor) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16.0, right: 16.0),
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        children: [
          Center(
            child: Text(
              'RESULT',
              style: TextStyle(
                color: textColor,
                fontWeight: FontWeight.bold,
                decoration: TextDecoration.underline,
                letterSpacing: 1.2,
              ),
            ),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: Row(
              children: [
                Expanded(
                  child: _VideoTile(
                    index: 0,
                    controller: _videoControllers[0],
                    isDarkMode: _isDarkMode,
                    textColor: textColor,
                    onFullScreen: () =>
                        setState(() => _fullScreenCameraIndex = 0),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: _VideoTile(
                    index: 1,
                    controller: _videoControllers[1],
                    isDarkMode: _isDarkMode,
                    textColor: textColor,
                    onFullScreen: () =>
                        setState(() => _fullScreenCameraIndex = 1),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: _VideoTile(
                    index: 2,
                    controller: _videoControllers[2],
                    isDarkMode: _isDarkMode,
                    textColor: textColor,
                    onFullScreen: () =>
                        setState(() => _fullScreenCameraIndex = 2),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: _VideoTile(
                    index: 3,
                    controller: _videoControllers[3],
                    isDarkMode: _isDarkMode,
                    textColor: textColor,
                    onFullScreen: () =>
                        setState(() => _fullScreenCameraIndex = 3),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDecisionTiles(Color cardColor, Color textColor) {
    return Container(
      margin: const EdgeInsets.only(right: 16.0, bottom: 16.0),
      child: Column(
        children: [
          Center(
            child: Text(
              'DECISION',
              style: TextStyle(
                color: textColor,
                fontWeight: FontWeight.bold,
                decoration: TextDecoration.underline,
                letterSpacing: 1.2,
              ),
            ),
          ),
          const SizedBox(height: 16),
          _buildDecisionRow(
            'BOWLER - STANCE:',
            _bowlerDecision,
            cardColor,
            textColor,
          ),
          const SizedBox(height: 8),
          _buildDecisionRow('DELIVERY:', _ballDecision, cardColor, textColor),
          const SizedBox(height: 8),
          _buildDecisionRow('WICKET:', _wicketDecision, cardColor, textColor),
        ],
      ),
    );
  }

  Widget _buildDecisionRow(
    String title,
    String activeValue,
    Color cardColor,
    Color textColor,
  ) {
    String displayOutcome = activeValue.toUpperCase();
    Color outcomeColor = textColor;

    if (activeValue.toUpperCase() == 'LEGAL' ||
        activeValue.toUpperCase() == 'VALID' ||
        activeValue.toUpperCase() == 'NOT-HIT') {
      outcomeColor = Colors.green;
    } else if (activeValue.toUpperCase() == 'ILLEGAL' ||
        activeValue.toUpperCase() == 'INVALID' ||
        activeValue.toUpperCase() == 'HIT' ||
        activeValue.toUpperCase() == 'WIDE' ||
        activeValue.toUpperCase() == 'NOBALL' ||
        activeValue.toUpperCase() == 'NO-BALL') {
      outcomeColor = Colors.red;
    } else if (activeValue.toUpperCase() == 'NIL') {
      outcomeColor = Colors.grey;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 12.0, horizontal: 24.0),
      decoration: BoxDecoration(
        color: _isDarkMode ? const Color(0xFF2C2C2E) : Colors.grey[300],
        borderRadius: BorderRadius.circular(24),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            title,
            style: TextStyle(
              color: textColor,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(width: 8),
          if (displayOutcome.isNotEmpty)
            Text(
              displayOutcome,
              style: TextStyle(
                color: outcomeColor,
                fontWeight: FontWeight.bold,
                fontSize: 16,
                fontStyle: FontStyle.italic,
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildFullScreenOverlay() {
    return Container(
      color: Colors.black.withValues(alpha: 0.9),
      child: Stack(
        alignment: Alignment.center,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: 3 / 4,
              child: Video(
                controller: _videoControllers[_fullScreenCameraIndex!],
                fit: BoxFit.fitHeight,
              ),
            ),
          ),
          Positioned(
            top: 32,
            right: 32,
            child: IconButton(
              icon: const Icon(
                Icons.fullscreen_exit,
                color: Colors.white,
                size: 48,
              ),
              onPressed: () {
                setState(() => _fullScreenCameraIndex = null);
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _VideoTile extends StatefulWidget {
  final int index;
  final VideoController controller;
  final bool isDarkMode;
  final Color textColor;
  final VoidCallback onFullScreen;

  const _VideoTile({
    required this.index,
    required this.controller,
    required this.isDarkMode,
    required this.textColor,
    required this.onFullScreen,
  });

  @override
  State<_VideoTile> createState() => _VideoTileState();
}

class _VideoTileState extends State<_VideoTile> {
  bool _isHovering = false;

  @override
  Widget build(BuildContext context) {
    return Consumer<DeviceListProvider>(
      builder: (context, provider, _) {
        final devices = provider.devices;
        final deviceName = widget.index < devices.length
            ? devices[widget.index].displayName
            : 'Cam ${widget.index + 1}';

        return Center(
          child: AspectRatio(
            aspectRatio: 3 / 4,
            child: MouseRegion(
              onEnter: (_) => setState(() => _isHovering = true),
              onExit: (_) => setState(() => _isHovering = false),
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: widget.isDarkMode
                        ? const Color(0xFF3A3A3C)
                        : Colors.grey,
                    width: 2,
                  ),
                ),
                clipBehavior: Clip.antiAlias,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Video(
                      controller: widget.controller,
                      controls: NoVideoControls,
                    ),

                    if (_isHovering)
                      Container(
                        color: Colors.black45,
                        child: Center(
                          child: IconButton(
                            icon: const Icon(
                              Icons.fullscreen,
                              color: Colors.white,
                              size: 48,
                            ),
                            onPressed: widget.onFullScreen,
                          ),
                        ),
                      ),

                    Positioned(
                      left: 8,
                      bottom: 8,
                      right: 8,
                      child: Row(
                        children: [
                          Container(
                            width: 32,
                            height: 32,
                            decoration: const BoxDecoration(
                              color: Color(0xFF2C2C2E),
                              shape: BoxShape.circle,
                            ),
                            alignment: Alignment.center,
                            child: Text(
                              '${widget.index + 1}',
                              style: const TextStyle(color: Colors.white),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 8,
                              ),
                              decoration: BoxDecoration(
                                color: const Color(0xFF2C2C2E),
                                borderRadius: BorderRadius.circular(16),
                              ),
                              child: Center(
                                child: Text(
                                  deviceName,
                                  style: const TextStyle(color: Colors.white),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Container(
                            width: 32,
                            height: 32,
                            decoration: const BoxDecoration(
                              color: Color(0xFF2C2C2E),
                              shape: BoxShape.circle,
                            ),
                            child: IconButton(
                              padding: EdgeInsets.zero,
                              icon: const Icon(
                                Icons.pause,
                                color: Colors.white,
                                size: 16,
                              ),
                              onPressed: () {
                                widget.controller.player.playOrPause();
                              },
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
