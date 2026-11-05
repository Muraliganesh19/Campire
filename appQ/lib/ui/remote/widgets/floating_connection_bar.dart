import 'package:flutter/material.dart';

class NetworkingBar extends StatelessWidget {
  final bool isConnected;
  final TextEditingController ipController;
  final String connectedIp;
  final VoidCallback onConnect;

  const NetworkingBar({
    super.key,
    required this.isConnected,
    required this.ipController,
    required this.connectedIp,
    required this.onConnect,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        // Left Button/Pill
        GestureDetector(
          onTap: isConnected ? null : onConnect,
          child: Container(
            height: 48.0,
            padding: const EdgeInsets.symmetric(horizontal: 24),
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: const Color(0xFF1C1C1E),
              borderRadius: BorderRadius.circular(24),
            ),
            child: Text(
              isConnected ? 'connected' : 'connect',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 16,
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        // Right Field
        Expanded(
          child: Container(
            height: 48.0,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: const Color(0xFF1C1C1E),
              borderRadius: BorderRadius.circular(24),
            ),
            child: isConnected
                ? Text(
                    connectedIp,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                    textAlign: TextAlign.center,
                  )
                : TextField(
                    controller: ipController,
                    style: const TextStyle(color: Colors.white, fontSize: 16),
                    textAlign: TextAlign.center,
                    decoration: const InputDecoration(
                      hintText: 'Enter IP Address',
                      hintStyle: TextStyle(color: Colors.white54, fontSize: 16),
                      border: InputBorder.none,
                      isDense: true,
                    ),
                  ),
          ),
        ),
      ],
    );
  }
}
