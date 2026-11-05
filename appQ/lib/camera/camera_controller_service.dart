import 'dart:async';
import 'dart:io' as io;
import 'package:camera/camera.dart';
import 'package:campire/core/constants.dart';
import 'package:campire/core/logger.dart';
import 'package:campire/utils/permissions.dart';
import 'package:gal/gal.dart';
import 'package:intl/intl.dart';

/// Wraps the `camera` plugin lifecycle.
///
/// Designed for a "pre-warm" pattern: the camera is opened and held in a
/// previewing state ahead of time. When a recording trigger arrives, we
/// only flip [startVideoRecording] / [stopVideoRecording] — no full camera
/// init — to minimise per-device startup variance.
///
/// Surface camera errors via [errors] so the UI can show a message instead
/// of crashing.
class CameraControllerService {
  final String localDeviceId;
  CameraController? _controller;
  CameraDescription? _selectedCamera;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;
  bool _isRecording = false;
  bool _isExposureLocked = false;

  final StreamController<String> _errorController =
      StreamController<String>.broadcast();

  CameraControllerService({required this.localDeviceId});

  // ── Public getters ──────────────────────────────────────────────────────

  /// The underlying camera controller, if initialized.
  /// Use this to build a [CameraPreview] in the UI.
  CameraController? get controller => _controller;

  /// Whether the camera has been initialized and is previewing.
  bool get isInitialized => _isInitialized;

  /// Whether a recording is currently in progress.
  bool get isRecording => _isRecording;

  /// Whether exposure is currently locked.
  ///
  /// This state persists independently of recording start/stop — locking
  /// exposure is only changed when the user explicitly toggles it.
  bool get isExposureLocked => _isExposureLocked;

  /// Stream of human-readable error messages for the UI.
  Stream<String> get errors => _errorController.stream;

  // ── Initialize (pre-warm) ──────────────────────────────────────────────

  /// Requests permissions, selects the back camera, creates a
  /// [CameraController], and initialises it in preview-ready state.
  ///
  /// Does **not** start recording — the camera stays open and previewing,
  /// ready for an instant [startRecording] call.
  ///
  /// Returns `true` if initialisation succeeded, `false` otherwise.
  Future<bool> initialize() async {
    // 1. Request permissions.
    final hasPermission = await PermissionHelper.requestAllPermissions();
    if (!hasPermission) {
      const msg = 'Camera/microphone/gallery permissions denied';
      SyncLogger.error(msg);
      _errorController.add(msg);
      return false;
    }

    // 2. List available cameras.
    try {
      _cameras = await availableCameras();
    } catch (e, stack) {
      const msg = 'Failed to list cameras';
      SyncLogger.error(msg, e, stack);
      _errorController.add(msg);
      return false;
    }

    if (_cameras.isEmpty) {
      const msg = 'No cameras available on this device';
      SyncLogger.error(msg);
      _errorController.add(msg);
      return false;
    }

    // 3. Select the back (rear-facing) camera. Fall back to the first
    //    available camera if no back camera is found.
    final backCamera = _cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras.first,
    );
    
    _selectedCamera = backCamera;

    SyncLogger.info(
      'CameraControllerService: selected camera '
      '${backCamera.name} (${backCamera.lensDirection.name})',
    );

    // 4. Create and initialise the controller.
    //    Try with the configured resolution + fps first; fall back to
    //    defaults if the device rejects the combination.
    if (!await _tryInitializeController(
      backCamera,
      AppConstants.targetResolutionPreset,
      fps: AppConstants.targetFps,
    )) {
      SyncLogger.info(
        'CameraControllerService: retrying with default fps '
        '(${AppConstants.targetResolutionPreset.name}, no fps override)',
      );
      if (!await _tryInitializeController(
        backCamera,
        AppConstants.targetResolutionPreset,
      )) {
        return false;
      }
    }

    return true;
  }

  /// Re-initialises the camera with new resolution and fps settings.
  ///
  /// Disposes the current controller and creates a new one, bringing it
  /// back to the pre-warmed state. Re-applies exposure lock if it was active.
  /// If [isRecording] is true, this is a no-op.
  Future<bool> reinitialize(ResolutionPreset resolution, int fps) async {
    if (_isRecording) {
      const msg = 'Cannot change camera settings while recording';
      SyncLogger.info('CameraControllerService: WARNING - $msg');
      _errorController.add(msg);
      return false;
    }
    if (_selectedCamera == null) {
      const msg = 'No camera selected to reinitialize';
      SyncLogger.error(msg);
      _errorController.add(msg);
      return false;
    }

    final wasExposureLocked = _isExposureLocked;
    
    SyncLogger.info(
      'CameraControllerService: reinitializing with '
      '${resolution.name} @ ${fps}fps',
    );

    if (!await _tryInitializeController(
      _selectedCamera!,
      resolution,
      fps: fps,
    )) {
      SyncLogger.info(
        'CameraControllerService: reinitialize rejected combo, retrying with '
        'fallback (${ResolutionPreset.high.name}, no fps override)',
      );
      if (!await _tryInitializeController(
        _selectedCamera!,
        ResolutionPreset.high,
      )) {
        return false;
      }
    }

    if (wasExposureLocked) {
      await lockExposure();
    }

    return true;
  }

  /// Attempts to create and initialise a [CameraController] with the given
  /// [preset] and optional [fps]. Returns `true` on success.
  Future<bool> _tryInitializeController(
    CameraDescription camera,
    ResolutionPreset preset, {
    int? fps,
  }) async {
    _controller?.dispose();
    _controller = CameraController(
      camera,
      preset,
      enableAudio: true,
      fps: fps,
    );

    try {
      await _controller!.initialize();
      _isInitialized = true;

      // Log what CameraX actually negotiated — this is how you find out
      // what each real device actually delivers.
      final previewSize = _controller!.value.previewSize;
      SyncLogger.info(
        'CameraControllerService: camera pre-warmed and previewing — '
        'requested: ${preset.name}${fps != null ? ' @ ${fps}fps' : ''}, '
        'actual preview size: '
        '${previewSize?.width.toInt()}×${previewSize?.height.toInt()}',
      );
      // NOTE: The camera plugin does not expose the actual negotiated fps.
      // The previewSize is the best available proxy for what the device
      // chose. Verify actual recording fps from the video file properties
      // on-device.

      return true;
    } on CameraException catch (e) {
      final msg = 'Camera initialisation failed: ${e.description}';
      SyncLogger.error(msg, e);
      _errorController.add(msg);
      _controller = null;
      _isInitialized = false;
      return false;
    } catch (e, stack) {
      final msg = 'Unexpected error during camera initialisation '
          '(${e.runtimeType}: $e)';
      SyncLogger.error(msg, e, stack);
      _errorController.add(msg);
      _controller = null;
      _isInitialized = false;
      return false;
    }
  }

  // ── Exposure lock ──────────────────────────────────────────────────────

  /// Locks the camera's auto-exposure at its current value.
  ///
  /// This state persists across recording start/stop — only an explicit
  /// call to [unlockExposure] changes it back.
  Future<void> lockExposure() async {
    if (!_isInitialized || _controller == null) {
      SyncLogger.info(
        'CameraControllerService: cannot lock exposure — camera not initialised',
      );
      return;
    }

    try {
      await _controller!.setExposureMode(ExposureMode.locked);
      _isExposureLocked = true;
      SyncLogger.info('CameraControllerService: exposure locked');
    } on CameraException catch (e) {
      final msg = 'Failed to lock exposure: ${e.description}';
      SyncLogger.error(msg, e);
      _errorController.add(msg);
    } catch (e, stack) {
      final msg = 'Failed to lock exposure (${e.runtimeType}: $e)';
      SyncLogger.error(msg, e, stack);
      _errorController.add(msg);
    }
  }

  /// Unlocks the camera's exposure, returning to automatic mode.
  Future<void> unlockExposure() async {
    if (!_isInitialized || _controller == null) {
      SyncLogger.info(
        'CameraControllerService: cannot unlock exposure — camera not initialised',
      );
      return;
    }

    try {
      await _controller!.setExposureMode(ExposureMode.auto);
      _isExposureLocked = false;
      SyncLogger.info('CameraControllerService: exposure unlocked (auto)');
    } on CameraException catch (e) {
      final msg = 'Failed to unlock exposure: ${e.description}';
      SyncLogger.error(msg, e);
      _errorController.add(msg);
    } catch (e, stack) {
      final msg = 'Failed to unlock exposure (${e.runtimeType}: $e)';
      SyncLogger.error(msg, e, stack);
      _errorController.add(msg);
    }
  }

  // ── Recording ──────────────────────────────────────────────────────────

  /// Starts video recording on the already-initialised camera.
  ///
  /// This should be fast since the camera is pre-warmed — no heavy init.
  Future<void> startRecording() async {
    if (!_isInitialized || _controller == null) {
      const msg = 'Cannot start recording — camera not initialised';
      SyncLogger.error(msg);
      _errorController.add(msg);
      return;
    }
    if (_isRecording) {
      SyncLogger.info('CameraControllerService: already recording');
      return;
    }

    try {
      await _controller!.startVideoRecording();
      _isRecording = true;
      SyncLogger.info('CameraControllerService: recording started');
    } on CameraException catch (e) {
      final msg = 'Failed to start recording: ${e.description}';
      SyncLogger.error(msg, e);
      _errorController.add(msg);
    }
  }

  /// Stops recording and returns the saved video file, or `null` on error.
  Future<XFile?> stopRecording() async {
    if (!_isRecording || _controller == null) {
      SyncLogger.info('CameraControllerService: not recording — nothing to stop');
      return null;
    }

    try {
      final file = await _controller!.stopVideoRecording();
      _isRecording = false;
      SyncLogger.info('CameraControllerService: recording stopped → ${file.path}');

      // Rename and save to gallery
      try {
        final dir = io.File(file.path).parent;
        final timestamp = DateFormat('yyyyMMdd_HHmmss').format(DateTime.now());
        final newPath = '${dir.path}/campire_${localDeviceId}_$timestamp.mp4';
        final newFile = await io.File(file.path).copy(newPath);

        await Gal.putVideo(newFile.path, album: 'Campire');
        SyncLogger.info('CameraControllerService: video saved to gallery → ${newFile.path}');
        return XFile(newFile.path);
      } catch (saveError, stack) {
        final msg = 'Failed to save video to gallery: $saveError';
        SyncLogger.error(msg, saveError, stack);
        _errorController.add(msg);
        return file; // return the original file if save fails
      }
    } on CameraException catch (e) {
      final msg = 'Failed to stop recording: ${e.description}';
      SyncLogger.error(msg, e);
      _errorController.add(msg);
      _isRecording = false;
      return null;
    }
  }

  // ── Dispose ────────────────────────────────────────────────────────────

  /// Releases the camera cleanly. Call on app exit or role switch.
  Future<void> dispose() async {
    _isRecording = false;
    _isInitialized = false;
    _isExposureLocked = false;

    try {
      await _controller?.dispose();
    } catch (e) {
      SyncLogger.error('CameraControllerService: error during dispose', e);
    }
    _controller = null;
    await _errorController.close();
    SyncLogger.info('CameraControllerService: disposed');
  }
}
