import 'dart:io';
import 'package:campire/core/logger.dart';

class PipelineResult {
  final String ballDecision;
  final String bowlerDecision;
  final List<String> videoPaths;

  PipelineResult({
    required this.ballDecision,
    required this.bowlerDecision,
    required this.videoPaths,
  });
}

class MockPipelineService {
  Future<PipelineResult> analyze() async {
    SyncLogger.info('Starting mock pipeline analysis...');
    
    // Simulate processing time
    await Future.delayed(const Duration(seconds: 3));

    final tempPath = '${Directory.systemTemp.path}${Platform.pathSeparator}campire_mock';
    final mockDecisionFile = File('$tempPath${Platform.pathSeparator}mock_decision.txt');
    
    String ballDecision = 'UNKNOWN';
    String bowlerDecision = 'UNKNOWN';

    if (await mockDecisionFile.exists()) {
      final lines = await mockDecisionFile.readAsLines();
      for (var line in lines) {
        if (line.startsWith('ball_decision=')) {
          ballDecision = line.split('=')[1];
        } else if (line.startsWith('bowler_decision=')) {
          bowlerDecision = line.split('=')[1];
        }
      }
    } else {
      SyncLogger.error('mock_decision.txt not found at ${mockDecisionFile.path}');
    }

    List<String> videoPaths = [];
    for (int i = 1; i <= 4; i++) {
      videoPaths.add('https://flutter.github.io/assets-for-api-docs/assets/videos/butterfly.mp4');
    }

    SyncLogger.info('Mock pipeline analysis complete.');
    
    return PipelineResult(
      ballDecision: ballDecision,
      bowlerDecision: bowlerDecision,
      videoPaths: videoPaths,
    );
  }
}
