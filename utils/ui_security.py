"""
utils/ui_security.py
=====================
Browser-side hardening that Streamlit has no built-in option for.

WHY THIS NEEDS JAVASCRIPT AT ALL: st.text_input(type="password") already
masks what is DISPLAYED (dots instead of characters), but that is purely
visual -- the real value still sits in the underlying <input>'s DOM
value, and a browser's native copy/cut/paste and right-click-menu actions
operate on that real value, not on the dots. Streamlit itself exposes no
parameter to disable those actions, so the only way to block them is to
reach into the rendered page with JavaScript and stop those specific
browser events ourselves.

HOW THE JAVASCRIPT REACHES THE REAL PAGE: st.components.v1.html() renders
its HTML inside an <iframe>, sandboxed from the rest of the page BY
DEFAULT for most content -- except this iframe is served from the same
Streamlit server as the main app (same origin), and Streamlit does not
mark it with a restrictive `sandbox` attribute, so `window.parent.document`
is reachable from inside it. That is what lets this component's script
attach listeners to <input> elements that live OUTSIDE its own iframe, in
the app's real page.

WHY A MutationObserver, NOT A ONE-TIME SCAN: Streamlit reruns the whole
script on every interaction, so a password field on one page (e.g. Login)
is a completely different <input> element than a password field on
another page (e.g. Change Password) rendered on a later rerun -- there is
no single fixed set of password fields to scan once. block_password_
clipboard() is called once per script run (see app.py's main()), each
call re-scans the CURRENT page immediately and also installs a fresh
observer that keeps watching for any password field added afterward
during that same run (e.g. a field inside a still-collapsed st.expander
that has not been opened yet). The `data-clipboard-blocked` marker
prevents attaching duplicate listeners if the observer fires more than
once for the same element.

WHAT THIS DOES NOT DO: this is a UI deterrent, not a real security
boundary -- anyone with browser DevTools open can still read an <input>'s
value directly, and that is unavoidable for any client-side field
(nothing server-side can prevent it, short of not rendering the value in
the browser at all, which a login form obviously must). Its purpose is to
stop the casual, everyday case: someone glancing at a shared screen
copying a password with a normal right-click or Ctrl+C/Ctrl+V, not to
resist a determined attacker with developer tools.
"""

import streamlit.components.v1 as components

_BLOCK_CLIPBOARD_JS = """
<script>
(function () {
    function blockClipboard(input) {
        if (input.dataset.clipboardBlocked) return;
        input.dataset.clipboardBlocked = "1";
        ["copy", "cut", "paste", "contextmenu"].forEach(function (eventName) {
            input.addEventListener(eventName, function (event) {
                event.preventDefault();
            });
        });
    }

    function scan(root) {
        root.querySelectorAll('input[type="password"]').forEach(blockClipboard);
    }

    var parentDoc = window.parent.document;
    scan(parentDoc);

    var observer = new MutationObserver(function () { scan(parentDoc); });
    observer.observe(parentDoc.body, { childList: true, subtree: true });
})();
</script>
"""


def block_password_clipboard() -> None:
    """
    Disable copy, cut, paste, and the right-click context menu (which
    offers the same actions) on every password field on the current page,
    including ones added later this same script run. Call once per
    script run -- see module docstring for why one call is enough to
    cover every page.
    """
    components.html(_BLOCK_CLIPBOARD_JS, height=0, width=0)
