import 'package:flutter/material.dart';
import 'package:campire/models/device_info.dart';
import 'package:campire/ui/shared/connection_status_badge.dart';

/// Tile displaying the status of a single remote camera device.
class DeviceListTile extends StatelessWidget {
  final DeviceInfo device;
  final VoidCallback? onTrigger;

  const DeviceListTile({
    super.key,
    required this.device,
    this.onTrigger,
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: const Icon(Icons.videocam),
      title: Text(device.displayName),
      subtitle: Text(device.ipAddress),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          ConnectionStatusBadge(status: device.connectionStatus),
          const SizedBox(width: 8),
          if (device.connectionStatus != ConnectionStatus.disconnected)
            IconButton(
              icon: Icon(device.isRecording ? Icons.stop : Icons.camera_alt),
              color: device.isRecording ? Colors.red : null,
              onPressed: onTrigger,
            ),
        ],
      ),
    );
  }
}
