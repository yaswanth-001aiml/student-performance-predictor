import { useState } from "react";
import api from "./api";
import "./App.css";

function App() {
  const [hours, setHours] = useState("");
  const [attendance, setAttendance] = useState("");
  const [sleep, setSleep] = useState("");
  const [previous, setPrevious] = useState("");
  const [tutoring, setTutoring] = useState("");

  const [score, setScore] = useState("");
  const [grade, setGrade] = useState("");
  const [status, setStatus] = useState("");

  const predict = async () => {
    try {
      const res = await api.post("/predict", {
        Hours_Studied: Number(hours),
        Attendance: Number(attendance),
        Sleep_Hours: Number(sleep),
        Previous_Scores: Number(previous),
        Tutoring_Sessions: Number(tutoring),
      });

      setScore(res.data.predicted_score);
      setGrade(res.data.grade);
      setStatus(res.data.status);
    } catch (err) {
      alert("Prediction failed!");
      console.log(err);
    }
  };

  return (
    <div className="container">
      <h1>Student Performance Predictor</h1>

      <input
        type="number"
        placeholder="Hours Studied"
        value={hours}
        onChange={(e) => setHours(e.target.value)}
      />

      <input
        type="number"
        placeholder="Attendance"
        value={attendance}
        onChange={(e) => setAttendance(e.target.value)}
      />

      <input
        type="number"
        placeholder="Sleep Hours"
        value={sleep}
        onChange={(e) => setSleep(e.target.value)}
      />

      <input
        type="number"
        placeholder="Previous Scores"
        value={previous}
        onChange={(e) => setPrevious(e.target.value)}
      />

      <input
        type="number"
        placeholder="Tutoring Sessions"
        value={tutoring}
        onChange={(e) => setTutoring(e.target.value)}
      />

      <button onClick={predict}>Predict</button>

      {score !== "" && (
        <div className="result">
          <h2>Predicted Score: {score}</h2>
          <h3>Grade: {grade}</h3>
          <h3>Status: {status}</h3>
        </div>
      )}
    </div>
  );
}

export default App;