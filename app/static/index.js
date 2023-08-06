var ws_conns = [];

function _arrayBufferToBase64(buffer) {
  var binary = "";
  var bytes = new Uint8Array(buffer);
  var len = bytes.byteLength;
  for (var i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

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
  var row = table.insertRow(-1);

  // Insert new cells (<td> elements) at the 1st and 2nd position of the "new" <tr> element:
  var cell0 = row.insertCell(0);
  var cell1 = row.insertCell(1);
  var cell2 = row.insertCell(2);
  var cell3 = row.insertCell(3);

  var ws = new WebSocket(
    `ws://localhost:8000/ws?user_id=${userIdInput.value}&conn_url=${connUrlInput.value}`
  );
  ws.binaryType = "arraybuffer";
  ws_conns.push(ws);
  // Add some text to the new cells:
  cell0.innerHTML = ws_conns.length;
  cell1.innerHTML = userIdInput.value;
  cell2.innerHTML = connUrlInput.value;
  cell3.innerHTML = "";
  cell3.innerHTML = `<span class="badge text-bg-success">Not Fall</span>`;
  ws.onmessage = function (event) {
    // var body = JSON.parse(event.data);
    // cell3.innerHTML = `<span class="badge ${
    //   body.state ? "text-bg-success" : "text-bg-danger"
    // }">${body.message}</span>`;

    cell3.innerHTML = `<span class="badge text-bg-danger">Fall</span>`;
    var imageContainer = document.getElementById("image-container");
    imageContainer.className = "container";

    showImage(event.data);
    // var content = document.createTextNode(body.message)
    // results.value = content
  };

  userIdInput.value = "";
  connUrlInput.value = "";
}
