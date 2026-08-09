import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:campire/state/app_role_provider.dart';
import 'package:campire/ui/role_select_screen.dart';
import 'package:campire/ui/controller/controller_home_screen.dart';
import 'package:campire/ui/remote/remote_home_screen.dart';

class CampireApp extends StatelessWidget {
  const CampireApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Campire Sync Camera',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: Consumer<AppRoleProvider>(
        builder: (context, roleProvider, _) {
          switch (roleProvider.role) {
            case AppRole.controller:
              return const ControllerHomeScreen();
            case AppRole.remote:
              return const RemoteHomeScreen();
            case AppRole.none:
              return const RoleSelectScreen();
          }
        },
      ),
    );
  }
}
