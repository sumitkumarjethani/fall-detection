var ws_conns = [];
var selectedWebSocket = null;

function showImage(data) {
  var img = document.getElementById("fall-image");
  img.src = "data:image/png;base64, " + data;
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

  var ws = new WebSocket(
    `ws://localhost:8000/ws?user_id=${userIdInput.value}&conn_url=${connUrlInput.value}`
  );

  ws.binaryType = "arraybuffer";
  ws_conns.push(ws);

  // Add some text to the new cells:
  cell0.innerHTML = ws_conns.length;
  cell1.innerHTML = userIdInput.value;
  cell2.innerHTML = connUrlInput.value;

  ws.onmessage = function (event) {
    var imageContainer = document.getElementById("image-container");
    imageContainer.className = "container";
    console.log(event);
    showImage(event.data);
  };
  userIdInput.value = "";
  connUrlInput.value = "";
}
