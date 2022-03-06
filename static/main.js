document.querySelector(".button").addEventListener("click", async()=>{
    const Http = new XMLHttpRequest();
    const url='http://127.0.0.1:5000/check/';
    Http.open("GET", url);
    Http.send();
    Http.onload = function() {
        console.log(Http.response);
        var img = document.createElement('img');
        img.src = '/static/final_output.png';
        document.querySelector('.popup').appendChild(img);
        document.querySelector("h2").innerHTML = "Done!"
        document.querySelector(".lds-ripple").style.display="none";
    };
    // await new Promise(resolve => setTimeout(resolve, 60000));
    
});