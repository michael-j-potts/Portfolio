const { PassThrough } = require('stream');

document.onkeydown = updateKey;
document.onkeyup = resetKey;

var server_port = 65432;
var server_addr = "192.168.0.14";   // the IP address of your Raspberry PI
var updata
var uptemp
var upbat
var upspeed

function client(){

    const net = require('net');
    var input = document.getElementById("message").value;

    const client = net.createConnection({ port: server_port, host: server_addr }, () => {
        // 'connect' listener.
        console.log('connected to server!');
        // send the message
        client.write(`${input}\r\n`);
    });

    // get the data from the server
    client.on('data', (data) => {
        document.getElementById("bluetooth").innerHTML = data;
        console.log(data.toString());
        client.end();
        client.destroy();
    });

    client.on('end', () => {
        console.log('disconnected from server');
    });
}


function updateCam() {
    const net = require('net');

    const client = net.createConnection({ port: server_port, host: server_addr }, () => {
        // 'connect' listener.
        console.log('connected to server!');
        client.write(`cam\r\n`);
    });

    //get the temp from the server
    client.on('data', (data) => {
        

        document.getElementById("cam").innerHTML = data;
        client.end();
        client.destroy();
    });

    client.on('end', () => {
        console.log('disconnected from server');
    });
}


function updateTemp(){
    //if (temp === 1) { break; }
    const net = require('net');

    const client = net.createConnection({ port: server_port, host: server_addr }, () => {
        // 'connect' listener.
        console.log('connected to server!');
        client.write(`temp\r\n`);
    });

    //get the temp from the server
    client.on('data', (data) => {
        document.getElementById("temp").innerHTML = data;
        console.log(data.toString());
        client.end();
        client.destroy();
    });

    client.on('end', () => {
        console.log('disconnected from server');
    });
}


function updateBat(){
    const net = require('net');

    const client = net.createConnection({ port: server_port, host: server_addr }, () => {
        // 'connect' listener.
        console.log('connected to server!');
        client.write(`bat\r\n`);
    });

    //get the temp from the server
    client.on('data', (data) => {
        var bat = Math.round(((parseFloat(data) - 5.25)/ 3.15) * 100);
        document.getElementById("bat").innerHTML = bat + "% remaining";
        console.log(data.toString());
        client.end();
        client.destroy();
    });

    client.on('end', () => {
        console.log('disconnected from server');
    });
}


// for detecting which key is been pressed w,a,s,d
function updateKey(e) {
    e = e || window.event;

    if (e.keyCode == '87') {
        // up (w)
        document.getElementById("upArrow").style.color = "green";

        const net = require('net');
        const client = net.createConnection({ port: server_port, host: server_addr }, () => {
            // 'connect' listener.
            console.log('connected to server!');
            // send the message
            client.write(`87\r\n`);
        });
        client.on('data', (data) => {
            document.getElementById("bluetooth").innerHTML = data;
            console.log(data.toString());
            client.end();
            client.destroy();
        });
    }
    else if (e.keyCode == '83') {
        // down (s)
        document.getElementById("downArrow").style.color = "green";

        const net = require('net');
        const client = net.createConnection({ port: server_port, host: server_addr }, () => {
            // 'connect' listener.
            console.log('connected to server!');
            // send the message
            client.write(`83\r\n`);
        });
        client.on('data', (data) => {
            document.getElementById("bluetooth").innerHTML = data;
            console.log(data.toString());
            client.end();
            client.destroy();
        });
    }
    else if (e.keyCode == '65') {
        // left (a)
        document.getElementById("leftArrow").style.color = "green";

        const net = require('net');
        const client = net.createConnection({ port: server_port, host: server_addr }, () => {
            // 'connect' listener.
            console.log('connected to server!');
            // send the message
            client.write(`65\r\n`);
        });
        client.on('data', (data) => {
            document.getElementById("bluetooth").innerHTML = data;
            console.log(data.toString());
            client.end();
            client.destroy();
        });
    }
    else if (e.keyCode == '68') {
        // right (d)
        document.getElementById("rightArrow").style.color = "green";

        const net = require('net');
        const client = net.createConnection({ port: server_port, host: server_addr }, () => {
            // 'connect' listener.
            console.log('connected to server!');
            // send the message
            client.write(`68\r\n`);
        });
        client.on('data', (data) => {
            document.getElementById("bluetooth").innerHTML = data;
            console.log(data.toString());
            client.end();
            client.destroy();
        });
    }
    else if (e.keyCode == '69') {
        // power up 10

        const net = require('net');
        const client = net.createConnection({ port: server_port, host: server_addr }, () => {
            // 'connect' listener.
            console.log('connected to server!');
            // send the message
            client.write(`69\r\n`);
        });
        client.on('data', (data) => {
            document.getElementById("bluetooth").innerHTML = data;
            console.log(data.toString());
            client.end();
            client.destroy();
        });
    }
    else if (e.keyCode == '81') {
        // power down 10

        const net = require('net');
        const client = net.createConnection({ port: server_port, host: server_addr }, () => {
            // 'connect' listener.
            console.log('connected to server!');
            // send the message
            client.write(`81\r\n`);
        });
        client.on('data', (data) => {
            document.getElementById("bluetooth").innerHTML = data;
            console.log(data.toString());
            client.end();
            client.destroy();
        });
    }
}

// reset the key to the start state
function resetKey(e) {
    e = e || window.event;

    document.getElementById("upArrow").style.color = "grey";
    document.getElementById("downArrow").style.color = "grey";
    document.getElementById("leftArrow").style.color = "grey";
    document.getElementById("rightArrow").style.color = "grey";

    const net = require('net');
    const client = net.createConnection({ port: server_port, host: server_addr }, () => {
        // 'connect' listener.
        console.log('connected to server!');
        // send the message
        client.write(`0\r\n`);
    });
}


// update data for every 50ms
function update_data(){
    updata = setInterval(function(){
        // get image from python server
        client();
    }, 1000);
}

function update_temp(){
    updateTemp();
    uptemp = setInterval(function(){
        // get image from python server
        updateTemp();
    }, 5000);
}

function update_bat(){
    updateBat();
    upbat = setInterval(function(){
        // get image from python server
        updateBat();
    }, 10000);
}

function update_cam(){
    upspeed = setInterval(function(){
        // get image from python server
        updateCam();
    }, 33.3333);
}

function stop_update(){
    clearInterval(updata);
}

function stop_temp(){
    clearInterval(uptemp);
}

function stop_bat(){
    clearInterval(upbat);
}

function stop_cam(){
    clearInterval(upspeed);
}
