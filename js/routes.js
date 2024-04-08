const title = "SPA Routing Tutorial"
const width = window.innerWidth;
const height = window.innerHeight;

let routes = {
  404:
  {
    site: "/sites/404.html",
    backgroundUri: "../img/404.jpg"
  },
  "/":
  {
    site: "/sites/home.html",
    backgroundUri: "../img/landing.jpg"
  },
  "/about":
  {
    site: "/sites/about.html",
    backgroundUri: "../img/background2.jpg"
  },
  "/contact":
  {
    site: "/sites/contact.html",
    backgroundUri: "../img/background2.jpg"
  },
  "/projects":
  {
    site: "/sites/projects.html",
    backgroundUri: "../img/background2.jpg"
  },
  "/model":
  {
    site: "/sites/predict_face.html",
    backgroundUri: "../img/background2.jpg"
  },
  "/canvas":
  {
    site: "/sites/canvas.html",
    backgroundUri: ""
  },
  "/cube":
  {
    site: "/sites/cube.html",
    backgroundUri: ""
  },
  "/todo":
  {
    site: "/sites/todo.html",
    backgroundUri: ""
  },
};

const route = (e) => {
  e.preventDefault();
  const href = e.target.getAttribute('href');
  window.history.pushState({}, "", href);
  handleRoute();
};

const handleRoute = async () => {
  const path = window.location.pathname;
  const route = routes[path] || routes[404];
  try {
    const html = await fetch(route.site).then((response) => {
      if (!response.ok) throw new Error(`Page not found: ${route}`);
      return response.text();
    });

    document.getElementById("templates").innerHTML = html;

    if (path === '/canvas') {
      import('/js/model/model.js').then((module) => {
        new module.default('canvas');
      }).catch(err => console.error('Failed to load the canvas app module:', err));

    } else if (path === '/todo') {
      document.getElementById("btn").addEventListener("click", (e) => {
        console.log("clicked");
      });
      const input = document.getElementById('todo-input')
      input.addEventListener('input', e => {
        const inputValue = parseInt(e.target.value, 10);
        if (!isNaN(inputValue) && inputValue.toString() === e.target.value && inputValue === 1234) {
          // console.log(false)
          console.log(typeof e.target.value)
        } else (
          console.log("wrong password, try again.")
        )
      })

    } else if (path === '/' || path === '/home') {
      document.querySelectorAll('nav a').forEach(link => {
        link.classList.remove('link');
        link.classList.add('home-link');
      });
      document.querySelector("#content").classList.remove("content");
      document.querySelector("#content").classList.add("content-home");
      // document.querySelector("strong h1").classList.remove("header-border-bottom")
      // document.querySelector("strong h1").classList.add("header-border-bottom-home")
      // document.getElementById("bg").style.backgroundColor = "rgba(0, 0, 0, 0.25)";
      // document.querySelector("strong h1").style.borderBottom = "3px solid #3ba171"

    } else {
      document.querySelectorAll('nav a').forEach(link => {
        link.classList.remove('home-link');
        link.classList.add('link');
      });
      document.querySelector("#content").classList.remove("content-home");
      document.querySelector("#content").classList.add("content");
      // document.querySelector("strong h1").classList.remove("header-border-bottom-home")
      // document.querySelector("strong h1").classList.add("header-border-bottom")
      // document.getElementById("bg").style.backgroundColor = "rgba(0, 0, 0, 0.8)";
    }

    document.title = `${title} | Welcome To The ${path} Page`

  } catch (error) {
    console.error("Error loading the page: ", error);
  }
};

window.onpopstate = handleRoute;
(document.addEventListener("DOMContentLoaded", () => {
  handleRoute();
  document.querySelectorAll('nav a').forEach(link => {
    link.addEventListener('click', route);
  });

  const getCoookieData = () => {
    const cookies = document.cookie.split(';');
    cookies ? (
      cookies.forEach((cookie) => {
        const [name, value] = cookie.split('=');
        console.log(`name: ${name}\ncookie: ${value}\n`);
      })
    ) : console.log("No cookies available.")
  };

  getCoookieData();
}))

