async function setMeeting(summary) {
    const response = await fetch("http://127.0.0.1:8000/calendar/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ summary: summary, description: "Scheduled via chatbot" }),
    });

    const data = await response.json();
    console.log(data);
    return data.message || "Error creating meeting";
}

// Example usage:
setMeeting("Project Discussion").then(response => console.log(response));


function fetchMeetings() {
    fetch("http://127.0.0.1:8000/meetings/voice")
    .then(response => response.json())
    .then(data => {
        let textToSpeak = data.voice;
        console.log(textToSpeak); // Debugging

        // ✅ Use Browser Speech Synthesis
        let speech = new SpeechSynthesisUtterance(textToSpeak);
        speech.lang = "en-US";
        speech.rate = 1.0;
        speech.pitch = 1.0;
        window.speechSynthesis.speak(speech);
    })
    .catch(error => console.error("Error fetching meetings:", error));
}
