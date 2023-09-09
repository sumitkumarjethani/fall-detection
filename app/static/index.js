var ws_conns = [];
var selectedWebSocket = null;

function showImage(data) {
  var img = document.getElementById("fall-image");
  img.src = "data:image/png;base64, " + data;
}

function connectWebSocket(event) {
  event.preventDefault();

  var userEmailInput = document.getElementById("userEmailInput");
  var connUrlInput = document.getElementById("connUrlInput");
  console.log(
    `connecting to ws: ${userEmailInput.value} - ${connUrlInput.value}`
  );

  var table = document.getElementById("results-table");

  // Create an empty <tr> element and add it to the 1st position of the table:
  var row = table.insertRow(ws_conns.length + 1);

  // Insert new cells (<td> elements) at the 1st and 2nd position of the "new" <tr> element:
  var cell0 = row.insertCell(0);
  var cell1 = row.insertCell(1);
  var cell2 = row.insertCell(2);

  var ws = new WebSocket(
    `ws://localhost:8000/ws?user_email=${userEmailInput.value}&conn_url=${connUrlInput.value}`
  );

  ws.binaryType = "arraybuffer";
  ws_conns.push(ws);

  // Add some text to the new cells:
  cell0.innerHTML = ws_conns.length;
  cell1.innerHTML = userEmailInput.value;

  ws.onmessage = function (event) {
    const response = JSON.parse(event.data);
    console.log(response);

    cell2.innerHTML = (response.detection === 1) ? "Fall" : "No Fall"
    
    var imageContainer = document.getElementById("image-container");
    imageContainer.className = "container";
    showImage(response.image);
  };
  userEmailInput.value = "";
  connUrlInput.value = "";
}
