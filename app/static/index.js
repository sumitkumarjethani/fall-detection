var ws_conns = [];
var selectedWebSocket = null;

function showImage(data) {
  var img = document.getElementById("fall-image");
  img.src = "data:image/png;base64, " + data;
}

function showStream(idx) {
  console.log("Websocket: " + idx)
  console.log(ws_conns)

  if (idx >= 0 && idx < ws_conns.length) {
    selectedWebSocket = ws_conns[idx];
  } else {
    selectedWebSocket = null
  }
}

function connectWebSocket(event) {
  event.preventDefault();

  var userIdInput = document.getElementById("userIdInput");
  var connUrlInput = document.getElementById("connUrlInput");
  console.log(`connecting to ws: ${userIdInput.value} - ${connUrlInput.value}`);


  var table = document.getElementById("results-table");

  // Create an empty <tr> element and add it to the 1st position of the table:
  var row = table.insertRow(ws_conns.length + 1);

  // Insert new cells (<td> elements) at the 1st and 2nd position of the "new" <tr> element:
  var cell0 = row.insertCell(0);
  var cell1 = row.insertCell(1);
  var cell2 = row.insertCell(2);
  var cell3 = row.insertCell(3);
  var cell4 = row.insertCell(4);

  var ws = new WebSocket(
    `ws://localhost:8000/ws?user_id=${userIdInput.value}&conn_url=${connUrlInput.value}`
  );

  ws.binaryType="arraybuffer"
  ws_conns.push(ws);

  // Add some text to the new cells:
  cell0.innerHTML = ws_conns.length;
  cell1.innerHTML = userIdInput.value;
  cell2.innerHTML = connUrlInput.value;
  cell3.innerHTML = `<span class="badge text-bg-success">Not Fall</span>`;
  cell4.innerHTML = `<button class="btn btn-primary show-image-btn" onclick="showStream(${ws_conns.length-1})">Show stream</button>`;

  ws.onmessage = function (event) {
    if (selectedWebSocket === ws) {
      cell3.innerHTML = `<span class="badge text-bg-danger">Fall</span>`;
      var imageContainer = document.getElementById("image-container");
      //imageContainer.className = "container";
      //showImage(JSON.parse(event.data).image);
      console.log(JSON.parse(event.data).message);
    }
  };
  userIdInput.value = "";
  connUrlInput.value = "";
}
