import 'package:flutter/foundation.dart';

/// The role this device plays in the sync camera system.
enum AppRole {
  /// No role selected yet — show role selection screen.
  none,

  /// This device triggers recording on all remotes.
  controller,

  /// This device records on the controller's command.
  remote,
}

/// Provider that manages the active role of this device.
///
/// The user selects their role (Controller or Remote) once at launch.
/// Changing the role tears down any active services and re-initialises
/// for the new role.
class AppRoleProvider extends ChangeNotifier {
  AppRole _role = AppRole.none;

  AppRole get role => _role;

  /// Sets the active role for the device.
  ///
  /// Service initialisation / teardown is handled at the widget level
  /// (in the home screens) because services need access to providers
  /// that are only available via the widget tree's BuildContext.
  void setRole(AppRole newRole) {
    if (_role == newRole) return;
    _role = newRole;
    notifyListeners();
  }
}
