import 'package:flutter/material.dart';
import 'package:campire/models/device_info.dart';

/// A shared badge component indicating the connection/sync status.
class ConnectionStatusBadge extends StatelessWidget {
  final ConnectionStatus status;

  const ConnectionStatusBadge({
    super.key,
    required this.status,
  });

  @override
  Widget build(BuildContext context) {
    Color color;
    String text;

    switch (status) {
      case ConnectionStatus.disconnected:
        color = Colors.red;
        text = 'Disconnected';
        break;
      case ConnectionStatus.connecting:
        color = Colors.orange;
        text = 'Connecting';
        break;
      case ConnectionStatus.connected:
        color = Colors.blue;
        text = 'Connected';
        break;
      case ConnectionStatus.synced:
        color = Colors.green;
        text = 'Synced';
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        // ignore: deprecated_member_use
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color, width: 1),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }
}
