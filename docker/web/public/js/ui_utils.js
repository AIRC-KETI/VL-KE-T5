// function - overlay

function overlay_off() {
    document.getElementById("overlay").style.display = "none";
}

function overlay_on(el) {
    document.getElementById("overlay").style.display = "block";
    const overlay_img = document.getElementById("overlay_img");
    overlay_img.src = el.src;
}

// function - create image elements
function create_single_img_elem(element) {
    const img_elem = document.createElement('img');
    img_elem.classList.add('zoom');
    img_elem.addEventListener("click", function () { overlay_on(this); }, false);
    img_elem.src = element.image_url;
    return img_elem;
}




