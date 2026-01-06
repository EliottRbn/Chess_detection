import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import { useRouter } from 'expo-router';

import { ChessBoard2D } from '../../src/components/ChessBoard2D';
import { BoardState } from '../../src/types';
import { parseFEN } from '../../src/utils/fenParser';

export interface HistoryEntry {
  id: string;
  fen: string;
  bestMove: string | null;
  date: string;
  turn: 'w' | 'b';
}

const HISTORY_KEY = '@chess_history';

export async function saveToHistory(entry: Omit<HistoryEntry, 'id' | 'date'>) {
  try {
    const existing = await AsyncStorage.getItem(HISTORY_KEY);
    const history: HistoryEntry[] = existing ? JSON.parse(existing) : [];
    
    const newEntry: HistoryEntry = {
      ...entry,
      id: Date.now().toString(),
      date: new Date().toISOString(),
    };
    
    history.unshift(newEntry);
    // Keep only last 50 entries
    if (history.length > 50) history.pop();
    
    await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch (error) {
    console.error('Failed to save history:', error);
  }
}

export default function HistoryScreen() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [selectedEntry, setSelectedEntry] = useState<HistoryEntry | null>(null);
  const router = useRouter();

  // Auto-reload when tab is focused
  useFocusEffect(
    useCallback(() => {
      loadHistory();
    }, [])
  );

  const loadHistory = async () => {
    try {
      const data = await AsyncStorage.getItem(HISTORY_KEY);
      if (data) {
        setHistory(JSON.parse(data));
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const clearHistory = () => {
    Alert.alert(
      'Effacer l\'historique',
      'Voulez-vous vraiment supprimer tout l\'historique ?',
      [
        { text: 'Annuler', style: 'cancel' },
        {
          text: 'Effacer',
          style: 'destructive',
          onPress: async () => {
            await AsyncStorage.removeItem(HISTORY_KEY);
            setHistory([]);
            setSelectedEntry(null);
          },
        },
      ]
    );
  };

  const formatDate = (isoDate: string) => {
    const date = new Date(isoDate);
    return date.toLocaleDateString('fr-FR', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const parseBestMove = (move: string | null) => {
    if (!move || move.length < 4) return null;
    return { from: move.substring(0, 2), to: move.substring(2, 4) };
  };

  const renderItem = ({ item }: { item: HistoryEntry }) => {
    const isSelected = selectedEntry?.id === item.id;
    
    return (
      <TouchableOpacity
        style={[styles.historyItem, isSelected && styles.historyItemSelected]}
        onPress={() => setSelectedEntry(isSelected ? null : item)}
      >
        <View style={styles.historyItemHeader}>
          <Text style={styles.historyDate}>{formatDate(item.date)}</Text>
          <Text style={styles.historyTurn}>
            {item.turn === 'w' ? '♙ Blancs' : '♟ Noirs'}
          </Text>
        </View>
        {item.bestMove && (
          <View style={styles.historyMove}>
            <Ionicons name="bulb-outline" size={14} color="#fbbf24" />
            <Text style={styles.historyMoveText}>
              {item.bestMove.substring(0, 2)} → {item.bestMove.substring(2, 4)}
            </Text>
          </View>
        )}
      </TouchableOpacity>
    );
  };

  const moveData = selectedEntry ? parseBestMove(selectedEntry.bestMove) : null;
  const boardState: BoardState | null = selectedEntry 
    ? parseFEN(selectedEntry.fen) 
    : null;

  return (
    <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.container}>
      <SafeAreaView style={styles.safeArea} edges={['top']}>
        <View style={styles.header}>
          {selectedEntry ? (
            <TouchableOpacity 
              onPress={() => setSelectedEntry(null)} 
              style={styles.backButton}
            >
              <Ionicons name="arrow-back" size={22} color="#fff" />
              <Text style={styles.backButtonText}>Retour</Text>
            </TouchableOpacity>
          ) : (
            <Text style={styles.headerTitle}>📋 Historique</Text>
          )}
          {history.length > 0 && !selectedEntry && (
            <TouchableOpacity onPress={clearHistory} style={styles.clearButton}>
              <Ionicons name="trash-outline" size={20} color="#ef4444" />
            </TouchableOpacity>
          )}
        </View>

        {history.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="time-outline" size={60} color="#4b5563" />
            <Text style={styles.emptyText}>Aucune analyse</Text>
            <Text style={styles.emptySubtext}>
              Vos analyses apparaîtront ici
            </Text>
          </View>
        ) : (
          <View style={styles.content}>
            {/* Board Preview */}
            {selectedEntry && boardState && (
              <View style={styles.previewCard}>
                <ChessBoard2D
                  boardState={boardState}
                  size={200}
                  highlightFrom={moveData?.from}
                  highlightTo={moveData?.to}
                />
                <View style={styles.fenContainer}>
                  <Text style={styles.fenLabel}>FEN</Text>
                  <Text style={styles.fenText} numberOfLines={2}>
                    {selectedEntry.fen}
                  </Text>
                </View>
              </View>
            )}

            {/* History List */}
            <FlatList
              data={history}
              renderItem={renderItem}
              keyExtractor={(item) => item.id}
              contentContainerStyle={styles.listContent}
              showsVerticalScrollIndicator={false}
            />
          </View>
        )}
      </SafeAreaView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  safeArea: { flex: 1 },
  
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 15,
  },
  headerTitle: { fontSize: 22, fontWeight: '700', color: '#fff' },
  clearButton: { padding: 8 },
  backButton: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    gap: 6 
  },
  backButtonText: { 
    color: '#fff', 
    fontSize: 16, 
    fontWeight: '500' 
  },

  content: { flex: 1, paddingHorizontal: 20 },

  previewCard: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 15,
    marginBottom: 15,
    alignItems: 'center',
  },
  fenContainer: {
    marginTop: 12,
    width: '100%',
  },
  fenLabel: { color: '#9ca3af', fontSize: 11, marginBottom: 4 },
  fenText: { color: '#fff', fontSize: 11, fontFamily: 'monospace' },

  listContent: { paddingBottom: 20 },
  
  historyItem: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    padding: 15,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  historyItemSelected: {
    borderColor: '#6366f1',
    backgroundColor: 'rgba(99,102,241,0.15)',
  },
  historyItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  historyDate: { color: '#fff', fontSize: 14, fontWeight: '500' },
  historyTurn: { color: '#9ca3af', fontSize: 12 },
  historyMove: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
    gap: 6,
  },
  historyMoveText: { color: '#fbbf24', fontSize: 14, fontWeight: '600' },

  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyText: { color: '#fff', fontSize: 18, fontWeight: '600', marginTop: 15 },
  emptySubtext: { color: '#6b7280', fontSize: 14, marginTop: 5 },
});
