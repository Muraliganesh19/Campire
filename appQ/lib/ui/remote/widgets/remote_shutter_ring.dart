import 'package:flutter/material.dart';

class RemoteShutterRing extends StatelessWidget {
  final bool isRecording;

  const RemoteShutterRing({super.key, required this.isRecording});

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: 72,
      height: 72,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: isRecording ? Colors.transparent : Colors.white,
        border: Border.all(
          color: Colors.white,
          width: isRecording ? 4.0 : 0.0,
        ),
      ),
      child: Center(
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          width: isRecording ? 32 : 0,
          height: isRecording ? 32 : 0,
          decoration: const BoxDecoration(
            color: Color(0xFFFF3B30),
            shape: BoxShape.circle,
          ),
        ),
      ),
    );
  }
}
