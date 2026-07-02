import { useQuery } from "@tanstack/react-query";
import { fetchFeed } from "../api/feed";

export function useFeed() {
  return useQuery({ queryKey: ["feed"], queryFn: fetchFeed });
}
