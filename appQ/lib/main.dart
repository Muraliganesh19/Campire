import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:campire/app.dart';
import 'package:campire/state/app_role_provider.dart';
import 'package:campire/state/device_list_provider.dart';
import 'package:campire/state/session_state_provider.dart';
import 'package:campire/utils/permissions.dart';
import 'package:campire/core/logger.dart';
import 'package:media_kit/media_kit.dart';
import 'dart:io';
import 'package:flutter/foundation.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  try {
    if (!kIsWeb && Platform.isWindows) {
      MediaKit.ensureInitialized();
    }
  } catch (e) {
    debugPrint('MediaKit initialization failed: $e');
  }

  // Request camera + microphone permissions early so the user sees the
  // prompt before choosing a role.
  final granted = await PermissionHelper.requestAllPermissions();
  SyncLogger.info('Permissions granted: $granted');

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AppRoleProvider()),
        ChangeNotifierProvider(create: (_) => DeviceListProvider()),
        ChangeNotifierProvider(create: (_) => SessionStateProvider()),
      ],
      child: const CampireApp(),
    ),
  );
}
