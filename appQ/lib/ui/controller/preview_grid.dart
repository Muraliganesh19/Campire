import 'package:flutter/material.dart';

/// Grid component to render tiled live preview streams from all remote devices (v2).
class PreviewGrid extends StatelessWidget {
  const PreviewGrid({super.key});

  @override
  Widget build(BuildContext context) {
    // TODO: Implement RTSP/MJPEG preview feeds from remote devices
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.grid_on, size: 64, color: Colors.grey),
          const SizedBox(height: 16),
          Text(
            'Tiled Live Preview Grid (v2)',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey),
          ),
          const SizedBox(height: 8),
          const Text(
            'Previews from active remotes will appear here',
            style: TextStyle(color: Colors.grey),
          ),
        ],
      ),
    );
  }
}
