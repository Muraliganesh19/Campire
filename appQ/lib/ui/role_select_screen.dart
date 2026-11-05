import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:campire/state/app_role_provider.dart';
import 'package:campire/ui/about_screen.dart';
import 'package:campire/ui/remote/remote_home_screen.dart';
import 'package:campire/ui/windows_host/windows_host_screen.dart';

class RoleSelectScreen extends StatefulWidget {
  const RoleSelectScreen({super.key});

  @override
  State<RoleSelectScreen> createState() => _RoleSelectScreenState();
}

class _RoleSelectScreenState extends State<RoleSelectScreen> {
  final TextEditingController _deviceNameController = TextEditingController();

  @override
  void dispose() {
    _deviceNameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final bool showController = kIsWeb || !Platform.isAndroid;
    final bool showRemote =
        !kIsWeb && !Platform.isWindows && !Platform.isLinux;

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildHeroHeader(),
                    const SizedBox(height: 24),
                    if (showRemote) ...[
                      // Custom Device Name Field
                      Container(
                        margin: const EdgeInsets.only(bottom: 16),
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1C1C1E),
                          borderRadius: BorderRadius.circular(30),
                          border: Border.all(color: const Color(0xFF3A3A3C), width: 1),
                        ),
                        child: TextField(
                          controller: _deviceNameController,
                          style: const TextStyle(color: Colors.white, fontSize: 16),
                          decoration: const InputDecoration(
                            hintText: 'Enter Device Name (e.g., Left Baseline)',
                            hintStyle: TextStyle(color: Colors.white54, fontSize: 15),
                            border: InputBorder.none,
                            icon: Icon(Icons.badge_rounded, color: Color(0xFF8E8E93)),
                          ),
                        ),
                      ),
                      _buildPrimaryCard(
                        context: context,
                        title: 'Boot Remote Camera',
                        subtitle: 'Synchronized 1080p60 Feed',
                        icon: Icons.videocam_rounded,
                        role: AppRole.remote,
                      ),
                    ],
                    if (showController) ...[
                      const SizedBox(height: 16),
                      _buildPrimaryCard(
                        context: context,
                        title: 'Launch Host Controller',
                        subtitle: 'Command & Monitor Feed',
                        icon: Icons.settings_remote_rounded,
                        role: AppRole.controller,
                      ),
                    ],
                  ],
                ),
              ),
            ),
            // Bottom Centered About Button
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 24.0),
              child: _buildAboutButton(context),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeroHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
      decoration: BoxDecoration(
        color: const Color(0xFF1C1C1E),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF3A3A3C), width: 1),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'CAMPIRE',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w900,
                  fontSize: 28,
                  letterSpacing: 2.0,
                ),
              ),
              const SizedBox(height: 4),
              const Text(
                'Sports Officiating Network',
                style: TextStyle(
                  color: Color(0xFF8E8E93),
                  fontWeight: FontWeight.w500,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: const Color(0xFFFF8500).withValues(alpha: 0.2),
              border: Border.all(color: const Color(0xFFFF8500), width: 1),
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Text(
              'v1.0.1',
              style: TextStyle(
                color: Color(0xFFFF8500),
                fontWeight: FontWeight.bold,
                fontSize: 14,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPrimaryCard({
    required BuildContext context,
    required String title,
    required String subtitle,
    required IconData icon,
    required AppRole role,
  }) {
    return GestureDetector(
      onTap: () {
        context.read<AppRoleProvider>().setRole(role);
        if (role == AppRole.remote) {
          final deviceName = _deviceNameController.text.trim();
          Navigator.push(context, MaterialPageRoute(builder: (_) => RemoteHomeScreen(deviceName: deviceName)));
        } else if (role == AppRole.controller) {
          Navigator.push(context, MaterialPageRoute(builder: (_) => const WindowsHostScreen()));
        }
      },
      child: Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: const Color(0xFF1C1C1E),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: const Color(0xFF3A3A3C), width: 1),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: const BoxDecoration(
                    color: Color(0xFF2C2C2E),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(icon, color: Color(0xFFFF8500), size: 36),
                ),
                const Icon(Icons.arrow_forward_ios_rounded, color: Colors.white54, size: 20),
              ],
            ),
            const SizedBox(height: 32),
            Text(
              title,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 22,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              subtitle,
              style: const TextStyle(
                color: Color(0xFF8E8E93),
                fontWeight: FontWeight.w500,
                fontSize: 15,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAboutButton(BuildContext context) {
    return InkWell(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => const AboutScreen()),
        );
      },
      borderRadius: BorderRadius.circular(30),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        decoration: BoxDecoration(
          color: const Color(0xFF1C1C1E),
          borderRadius: BorderRadius.circular(30),
          border: Border.all(color: const Color(0xFF3A3A3C), width: 1),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: const [
            Icon(Icons.info_outline_rounded, color: Colors.white70, size: 20),
            SizedBox(width: 8),
            Text(
              'About Campire',
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
