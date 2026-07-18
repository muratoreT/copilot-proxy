/**
 * Lightweight collapsible JSON tree renderer.
 * Usage: renderJsonTree(containerElement, data)
 */

function renderJsonTree(container, data) {
    container.innerHTML = '';
    container.className = 'json-tree';
    container.appendChild(buildNode(data));
}

function buildNode(value, key = null) {
    const type = jsonType(value);
    const fragment = document.createDocumentFragment();

    const line = document.createElement('span');
    line.style.display = 'block';

    // Key prefix
    if (key !== null) {
        const keySpan = document.createElement('span');
        keySpan.className = 'json-key';
        keySpan.textContent = `"${key}": `;
        line.appendChild(keySpan);
    }

    if (type === 'object' || type === 'array') {
        const entries = type === 'object' ? Object.entries(value) : value.map((v, i) => [i, v]);
        const toggle = document.createElement('span');
        toggle.className = 'json-toggle';
        toggle.textContent = '\u25BC';
        line.appendChild(toggle);

        const brace = document.createElement('span');
        brace.textContent = type === 'object' ? '{' : '[';
        line.appendChild(brace);

        const childContainer = document.createElement('span');
        childContainer.className = 'json-children';

        entries.forEach(([k, v], idx) => {
            const indent = document.createTextNode('  ');
            childContainer.appendChild(indent);
            childContainer.appendChild(buildNode(v, type === 'object' ? k : null));
        });

        const closeLine = document.createElement('span');
        closeLine.style.display = 'block';
        closeLine.textContent = '  ' + (type === 'object' ? '}' : ']');
        childContainer.appendChild(closeLine);

        // Toggle behavior
        let expanded = true;
        let collapsedHint = null;

        toggle.addEventListener('click', () => {
            expanded = !expanded;
            childContainer.style.display = expanded ? '' : 'none';
            toggle.textContent = expanded ? '\u25BC' : '\u25B6';

            if (!expanded && !collapsedHint) {
                collapsedHint = document.createElement('span');
                collapsedHint.className = 'json-collapsed';
                collapsedHint.textContent = ` ${entries.length} ${type === 'object' ? 'keys' : 'items'}`;
                line.appendChild(collapsedHint);
            } else if (expanded && collapsedHint) {
                collapsedHint.remove();
                collapsedHint = null;
            }
        });

        fragment.appendChild(line);
        fragment.appendChild(childContainer);
    } else {
        // Primitive value
        const valSpan = document.createElement('span');
        valSpan.className = `json-${type}`;
        if (type === 'string') {
            valSpan.textContent = `"${value}"`;
        } else {
            valSpan.textContent = String(value);
        }
        line.appendChild(valSpan);
        fragment.appendChild(line);
    }

    return fragment;
}

function jsonType(value) {
    if (value === null) return 'null';
    if (Array.isArray(value)) return 'array';
    return typeof value;
}
