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

/**
 * Render body content split into a Properties table and collapsible sections
 * for arrays and large nested objects.
 * Usage: renderBodySections(containerElement, data)
 */
function renderBodySections(container, data) {
    container.innerHTML = '';

    if (data === null || data === undefined || data === '') {
        container.textContent = '(no body)';
        return;
    }

    const normalized = normalizeBodyData(data);
    const body = normalized.value;

    if (body === null || body === undefined || body === '') {
        container.textContent = '(no body)';
        return;
    }

    if (Array.isArray(body)) {
        appendCollapsibleSection(container, 'Items', body.length, body);
        return;
    }

    if (typeof body !== 'object') {
        appendRawBodySection(container, body, normalized.wasJsonString);
        return;
    }

    const scalars = [];
    const collapsibles = [];

    for (const [key, value] of Object.entries(body)) {
        const type = jsonType(value);

        if (type === 'array') {
            collapsibles.push({ key, value, label: capitalize(key), count: value.length });
        } else if (type === 'object') {
            const keyCount = Object.keys(value).length;
            if (keyCount > 3) {
                collapsibles.push({ key, value, label: capitalize(key), count: keyCount });
            } else {
                scalars.push({ key, value });
            }
        } else {
            scalars.push({ key, value });
        }
    }

    // Render Properties section (always visible)
    if (scalars.length > 0) {
        const propsSection = document.createElement('div');
        propsSection.className = 'properties-section';

        const propsHeading = document.createElement('h4');
        propsHeading.className = 'properties-heading';
        propsHeading.textContent = 'Properties';
        propsSection.appendChild(propsHeading);

        const propsTable = document.createElement('table');
        propsTable.className = 'properties-table';

        scalars.forEach(({ key, value }) => {
            const row = document.createElement('tr');
            const keyCell = document.createElement('td');
            keyCell.className = 'property-key';
            keyCell.textContent = key;

            const valCell = document.createElement('td');
            valCell.className = 'property-value';

            if (typeof value === 'object' && value !== null) {
                // Small nested object — render as compact JSON tree
                const jsonContainer = document.createElement('div');
                jsonContainer.className = 'json-tree';
                jsonContainer.appendChild(buildNode(value));
                valCell.appendChild(jsonContainer);
            } else if (typeof value === 'string') {
                // Truncate long strings for display
                if (value.length > 200) {
                    const shortText = document.createElement('span');
                    shortText.textContent = value.substring(0, 200) + '…';
                    shortText.title = value;
                    valCell.appendChild(shortText);
                } else {
                    valCell.textContent = value;
                }
            } else {
                valCell.textContent = String(value);
            }

            row.appendChild(keyCell);
            row.appendChild(valCell);
            propsTable.appendChild(row);
        });

        propsSection.appendChild(propsTable);
        container.appendChild(propsSection);
    }

    // Render collapsible sections for arrays and large objects
    collapsibles.forEach(({ label, count, value }) => {
        appendCollapsibleSection(container, label, count, value);
    });

    if (scalars.length === 0 && collapsibles.length === 0) {
        appendCollapsibleSection(container, 'Body', Object.keys(body).length, body);
    }
}

function normalizeBodyData(data) {
    if (typeof data !== 'string') {
        return { value: data, wasJsonString: false };
    }

    const trimmed = data.trim();
    if (!trimmed) {
        return { value: '', wasJsonString: false };
    }

    const looksLikeJson =
        (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
        (trimmed.startsWith('[') && trimmed.endsWith(']'));

    if (!looksLikeJson) {
        return { value: data, wasJsonString: false };
    }

    try {
        return { value: JSON.parse(trimmed), wasJsonString: true };
    } catch (_err) {
        return { value: data, wasJsonString: false };
    }
}

function appendCollapsibleSection(container, label, count, value) {
    const details = document.createElement('details');
    details.className = 'collapsible';

    const summary = document.createElement('summary');
    summary.textContent = `${label} (${count})`;
    details.appendChild(summary);

    const jsonContainer = document.createElement('div');
    jsonContainer.className = 'json-tree';
    jsonContainer.appendChild(buildNode(value));
    details.appendChild(jsonContainer);

    container.appendChild(details);
}

function appendRawBodySection(container, value, parsedFromJsonString) {
    const details = document.createElement('details');
    details.className = 'collapsible';

    const summary = document.createElement('summary');
    summary.textContent = parsedFromJsonString ? 'Body (parsed fallback)' : 'Body';
    details.appendChild(summary);

    const pre = document.createElement('pre');
    pre.className = 'json-tree';
    pre.textContent = String(value);
    details.appendChild(pre);

    container.appendChild(details);
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
