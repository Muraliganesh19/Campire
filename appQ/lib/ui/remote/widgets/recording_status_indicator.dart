import 'dart:ui';
import 'package:flutter/material.dart';

class RecordingStatusIndicator extends StatelessWidget {
  final bool isRecording;
  final Duration duration;

  const RecordingStatusIndicator({
    super.key,
    required this.isRecording,
    required this.duration,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          color: const Color(0xFF1C1C1E).withValues(alpha: 0.85),
          child: Text(
            isRecording ? _formatDuration(duration) : '00:00',
            style: const TextStyle(
              color: Color(0xFFFF3B30),
              fontSize: 16,
              fontWeight: FontWeight.bold,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
        ),
      ),
    );
  }

  String _formatDuration(Duration d) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final minutes = twoDigits(d.inMinutes.remainder(60));
    final seconds = twoDigits(d.inSeconds.remainder(60));
    return '$minutes:$seconds';
  }
}
