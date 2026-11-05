import 'package:flutter/material.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'About',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Campire Architecture',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Campire is a zero-internet multi-device synchronization engine designed for sports officiating. It leverages local WebSocket channels, NTP-based round-trip-time offset calculations, and scheduled future-execution timestamps to trigger video recordings across multiple disparate SoC devices.',
                style: TextStyle(
                  color: Color(0xFF8E8E93),
                  fontSize: 16,
                  height: 1.5,
                ),
              ),
              const SizedBox(height: 32),
              _buildSectionHeader('Key Features'),
              const SizedBox(height: 16),
              _buildBulletPoint('Multi-camera sports officiating tool'),
              _buildBulletPoint('Sub-frame WebSocket NTP synchronization'),
              _buildBulletPoint('Automated clip siphon'),
              const SizedBox(height: 32),
              _buildSectionHeader('System Specs'),
              const SizedBox(height: 16),
              _buildBulletPoint('6 Camera Nodes Capable'),
              _buildBulletPoint('HTTP Upload Siphon'),
              _buildBulletPoint('SHA-256 Hash Verified'),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Text(
      title,
      style: const TextStyle(
        color: Color(0xFFFF8500),
        fontSize: 16,
        fontWeight: FontWeight.bold,
        letterSpacing: 1.2,
      ),
    );
  }

  Widget _buildBulletPoint(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 6.0),
            child: Icon(Icons.circle, color: Color(0xFF8E8E93), size: 6),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
