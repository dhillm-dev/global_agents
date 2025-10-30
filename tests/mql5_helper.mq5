string GetDecision(string sym, string tf){
  string url = "https://global-agents.vercel.app/api/ta_decision";
  string body = "{\"symbol\":\"" + sym + "\",\"tf\":\"" + tf + "\"}";
  char result[];
  int res = WebRequest("POST", url, "application/json", body, result, 5000);
  if(res == 200){
    string json = CharArrayToString(result);
    if(StringFind(json, "\"action\":\"BUY\"") >= 0) return "BUY";
    if(StringFind(json, "\"action\":\"SELL\"") >= 0) return "SELL";
  }
  return "NONE";
}