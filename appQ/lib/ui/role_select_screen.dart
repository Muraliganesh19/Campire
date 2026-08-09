import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:campire/state/app_role_provider.dart';

/// Screen allowing the user to select their device's role on launch.
class RoleSelectScreen extends StatelessWidget {
  const RoleSelectScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Select Role'),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Welcome to Campire Sync Camera.\nChoose this device\'s role:',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 48),
              ElevatedButton.icon(
                onPressed: () {
                  context.read<AppRoleProvider>().setRole(AppRole.controller);
                },
                icon: const Icon(Icons.settings_remote),
                label: const Text('Controller Device'),
                style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
              ),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: () {
                  context.read<AppRoleProvider>().setRole(AppRole.remote);
                },
                icon: const Icon(Icons.camera_alt),
                label: const Text('Remote Camera'),
                style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
              ),
              const SizedBox(height: 48),
              const Text(
                'v1.0.1',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
