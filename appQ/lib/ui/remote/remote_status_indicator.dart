import 'package:flutter/material.dart';
import 'package:campire/state/session_state_provider.dart';

/// Screen component displaying the current session recording state on the
/// remote device — pairing, synced, recording, etc.
class RemoteStatusIndicator extends StatelessWidget {
  final SessionStatus status;

  const RemoteStatusIndicator({
    super.key,
    required this.status,
  });

  @override
  Widget build(BuildContext context) {
    final Color color;
    final String titleText;
    final String descText;

    switch (status) {
      case SessionStatus.idle:
        color = Colors.blueGrey;
        titleText = 'Waiting for Controller';
        descText = 'Listening for discovery broadcast...';
      case SessionStatus.awaitingSync:
        color = Colors.orange;
        titleText = 'Syncing';
        descText = 'Start command received — awaiting scheduled time...';
      case SessionStatus.recording:
        color = Colors.red;
        titleText = 'REC';
        descText = 'Synchronised recording in progress';
      case SessionStatus.error:
        color = Colors.redAccent;
        titleText = 'Error';
        descText = 'An error occurred during synchronisation.';
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        // ignore: deprecated_member_use
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color, width: 2),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (status == SessionStatus.recording)
                const Padding(
                  padding: EdgeInsets.only(right: 8.0),
                  child: Icon(Icons.circle, color: Colors.red, size: 16),
                ),
              Text(
                titleText.toUpperCase(),
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            descText,
            textAlign: TextAlign.center,
            style: const TextStyle(fontSize: 14),
          ),
        ],
      ),
    );
  }
}
