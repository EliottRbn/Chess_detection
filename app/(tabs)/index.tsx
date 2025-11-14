import { View, Text, StyleSheet } from "react-native";

export default function TabHomeScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Tab Home OK</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fff",
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
  },
});