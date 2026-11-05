import 'package:flutter/material.dart';
import 'package:campire/ui/role_select_screen.dart';

class CampireApp extends StatelessWidget {
  const CampireApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Campire Sync Camera',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const RoleSelectScreen(),
    );
  }
}
