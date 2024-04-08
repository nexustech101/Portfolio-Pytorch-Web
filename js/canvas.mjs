export default class DrawingApp {
  constructor() {
    this.canvas = document.querySelector("#canvas");
    this.clearBtn = document.querySelector("#clearBtn");
    this.undoBtn = document.querySelector("#undo");
    this.eraserBtn= document.querySelector("#undo");
    this.colorBtn = document.querySelectorAll("#changeColor");
    this.ctx = this.canvas.getContext("2d");
    this.initializeCanvas();

    this.color = "white";
    this.lineWidth = 2;
    this.cacheMemory = [];
    this.index = -1;

    this.mouse = {
      x: null,
      y: null,
      isDrawing: false,
    };

    this.attachEventListeners();
  }

  initializeCanvas() {
    this.canvas.height = window.innerHeight;
    this.canvas.width = window.innerWidth;
    this.canvas.style.backgroundColor = "#111111";
  }

  attachEventListeners() {
    this.canvas.addEventListener("mousedown", this.startDraw.bind(this), false);
    this.canvas.addEventListener("mousemove", this.draw.bind(this), false);
    this.canvas.addEventListener("mouseup", this.setActiveStatus.bind(this), false);
    this.canvas.addEventListener("mouseout", this.setActiveStatus.bind(this), false);
    this.clearBtn.addEventListener("click", this.clearCanvas.bind(this), false);
    this.undoBtn.addEventListener("click", this.undo.bind(this), false);
    this.eraserBtn.addEventListener("click", this.undo.bind(this), false);
    this.colorBtn.forEach(colorPicker => {
      colorPicker.addEventListener("click", (e) => this.changeColor(e.target), false);
    });
  }

  startDraw(e) {
    this.mouse.isDrawing = true;
    this.mouse.x = e.clientX;
    this.mouse.y = e.clientY;
    this.ctx.beginPath();
    this.ctx.moveTo(e.clientX - this.canvas.offsetLeft, e.clientY - this.canvas.offsetTop);
    this.ctx.stroke();
  }

  draw(e) {
    if (!this.mouse.isDrawing) return;
    this.ctx.lineTo(e.clientX - this.canvas.offsetLeft, e.clientY - this.canvas.offsetTop);
    this.ctx.strokeStyle = this.color;
    this.ctx.lineWidth = this.lineWidth;
    this.ctx.lineCap = "round";
    this.ctx.lineJoin = "round";
    this.ctx.stroke();
  }

  clearCanvas() {
    this.ctx.fillStyle = '#111111';
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.cacheMemory = [];
    this.index = -1;
  }

  setActiveStatus(e) {
    this.mouse.isDrawing = false;
    if (e.type !== "mouseout") {
      this.cacheMemory.push(this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height));
      this.index += 1;
    }
  }

  undo() {
    if (this.index <= 0) {
      this.clearCanvas();
    } else {
      this.index -= 1;
      this.cacheMemory.pop();
      this.ctx.putImageData(this.cacheMemory[this.index], 0, 0);
    }
  }

  changeColor(element) {
    this.color = element.style.backgroundColor;
  }
}
