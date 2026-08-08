(() => {
  const nav = document.querySelector(".site-nav");
  if (!nav) return;
  const siteVersion = "20260809-3";

  const pageFromLink = (link) => {
    const file = new URL(link.href, window.location.href).pathname.split("/").pop().toLowerCase();
    if (file === "blog.html") return "blog";
    if (file === "photography.html") return "photography";
    return "home";
  };

  let navigating = false;

  nav.addEventListener("click", (event) => {
    const link = event.target.closest("a");
    if (!link || navigating || event.defaultPrevented) return;
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

    const destination = new URL(link.href, window.location.href);
    const current = new URL(window.location.href);
    if (destination.origin !== current.origin) return;

    const sameDocument = destination.pathname === current.pathname;
    if (sameDocument) {
      event.preventDefault();
      return;
    }

    destination.searchParams.set("sitev", siteVersion);

    const targetPage = pageFromLink(link);
    const currentPage = document.body.dataset.page || "home";
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (targetPage === currentPage || reduceMotion) {
      event.preventDefault();
      window.location.assign(destination.href);
      return;
    }

    event.preventDefault();
    navigating = true;
    document.documentElement.dataset.navTarget = targetPage;

    let completed = false;
    const finish = () => {
      if (completed) return;
      completed = true;
      nav.removeEventListener("transitionend", onTransitionEnd);
      window.location.assign(destination.href);
    };
    const onTransitionEnd = (transitionEvent) => {
      if (
        transitionEvent.target === nav &&
        transitionEvent.pseudoElement === "::before" &&
        transitionEvent.propertyName === "transform"
      ) finish();
    };

    nav.addEventListener("transitionend", onTransitionEnd);
    window.setTimeout(finish, 280);
  });
})();
