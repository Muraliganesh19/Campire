import 'package:permission_handler/permission_handler.dart';
import 'package:campire/core/logger.dart';

import 'package:gal/gal.dart';

/// Helper class to request and check required system permissions.
///
/// The app needs camera and microphone permissions to record video.
/// Storage permissions are not needed on modern Android (scoped storage)
/// because we use [path_provider] to write to app-specific directories,
/// but we use [gal] to save to the gallery/albums, which requires access.
class PermissionHelper {
  /// Requests camera, microphone, and gallery permissions.
  ///
  /// Returns `true` if **all** are granted after the request.
  static Future<bool> requestAllPermissions() async {
    SyncLogger.info('PermissionHelper: requesting camera + microphone + gallery');

    final statuses = await [
      Permission.camera,
      Permission.microphone,
    ].request();

    final cameraGranted = statuses[Permission.camera]?.isGranted ?? false;
    final micGranted = statuses[Permission.microphone]?.isGranted ?? false;

    var galGranted = await Gal.hasAccess();
    if (!galGranted) {
      galGranted = await Gal.requestAccess();
    }

    SyncLogger.info(
      'PermissionHelper: camera=$cameraGranted, microphone=$micGranted, gallery=$galGranted',
    );

    return cameraGranted && micGranted && galGranted;
  }

  /// Checks whether camera, microphone, and gallery permissions are already granted.
  static Future<bool> hasPermissions() async {
    final camera = await Permission.camera.isGranted;
    final microphone = await Permission.microphone.isGranted;
    final gal = await Gal.hasAccess();
    return camera && microphone && gal;
  }
}
