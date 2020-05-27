// =================================================
//  CONSTANTS
// =================================================

var MODELS = {
    "vanilla":      {"dir": "final_vanilla_ddsp",      "label": "vanilla-DDSP"},
    "mel":          {"dir": "final_cyl",               "label": "mel-DDSP"},
    "cyl":          {"dir": "final_cyl2",              "label": "cylinder-DDSP"},
    "phase":        {"dir": "final_phase",             "label": "phase-DDSP"},
    "large":        {"dir": "final_large_gru",         "label": "large-DDSP"},
    "hnt":          {"dir": "final_hnt",               "label": "transient-DDSP"},
    "mini-vanilla": {"dir": "final_mini_vanilla_ddsp", "label": "vanilla-DDSP"},
    "mini-mel":     {"dir": "final_mini_cyl",          "label": "mel-DDSP"},
    "mini-cyl":     {"dir": "final_mini_cyl2",         "label": "cylinder-DDSP"},
    "mini-phase":   {"dir": "final_mini_phase",        "label": "phase-DDSP"},
    "mini-hnt":     {"dir": "final_mini_hnt",          "label": "transient-DDSP"}
};

// =================================================
//  FUNCTIONS
// =================================================

// Get file prefix
function getFilePrefix(experiment) {
    if (experiment === "control") {
        return "control_";
    } else {
        return "audio_"
    }
}

// Generate table head
function generateTableHead(table, columns) {
    let thead = table.createTHead();
    let row = thead.insertRow();
    let th = document.createElement("th");
    row.appendChild(th);
    for (let key of columns) {
      let th = document.createElement("th");
      let text = document.createTextNode(key);
      th.appendChild(text);
      row.appendChild(th);
    }
}

// Generate table row
function generateTableRow(table, model, dir, prefix, suffixes) {
    let row = table.insertRow();
    let cell = row.insertCell();
    let text = document.createTextNode(model);
    cell.appendChild(text);
    for (let suffix of suffixes) {
        let audio;
        if ( suffix.includes("transient") && !model.includes("transient") ) {
            audio = document.createTextNode("N/A");
        } else {
            audio = document.createElement("audio");
            let audioPath = dir + "/data/" + prefix + (suffix === "" ? "" : "_") + suffix + ".wav";
            audio.src = audioPath;
            audio.controls = "controls";
        }
        let cell = row.insertCell();
        cell.appendChild(audio);
    }
}

// Generate table
function generateTable(experiment, example, models, columns, suffixes) {
    let div = document.getElementById(experiment);
    let heading = document.createElement("h2");
    let headingText = document.createTextNode(example.label);
    heading.appendChild(headingText);
    let table = document.createElement("table");
    
    generateTableHead(table, columns);

    models.forEach(function(model) {
        let prefix = getFilePrefix(experiment) + example.id;
        let label = MODELS[model].label;
        let dir = "fig/" + MODELS[model].dir;
        generateTableRow(table, label, dir, prefix, suffixes);
    });

    div.appendChild(heading);
    div.appendChild(table);
}

// Generate tables
function generateTables(experiment, examples, models, columns, suffixes) {
    examples.forEach(function(example) {
        generateTable(experiment, example, models, columns, suffixes)
    });
}

// =================================================
//  GENERATE TABLES
// =================================================
var examples;
var models;
var columns;
var suffixes;

// Generate tables for Overfitting to a Small Dataset
examples = [
    {"id": "train_1", "label": "Train: Idling"},
    {"id": "train_3", "label": "Train: Mid-Range RPM"},
    {"id": "train_2", "label": "Train: High-Range RPM"}
];
models = ["mini-vanilla", "mini-mel", "mini-cyl", "mini-phase", "mini-hnt"];
columns = ["original", "reconstruction", "harmonic", "noise", "transients"];
suffixes = ["rec", "syn", "syn_additive", "syn_subtractive", "syn_transients"];

generateTables("overfit", examples, models, columns, suffixes);

// Generate tables for Reconstruction
examples = [
    {"id": "train_1",   "label": "Train: Idling"},
    {"id": "train_127", "label": "Train: Mid-Range RPM"},
    {"id": "train_78",  "label": "Train: High-Range RPM"},
    {"id": "test_1",    "label": "Test: Idling"},
    {"id": "test_42",   "label": "Test: Mid-Range RPM"},
    {"id": "test_49",   "label": "Test: High-Range RPM"}
];
models = ["vanilla", "mel", "cyl", "phase", "hnt", "large"];
columns = ["original", "reconstruction", "harmonic", "noise", "transients"];
suffixes = ["rec", "syn", "syn_additive", "syn_subtractive", "syn_transients"];

generateTables("reconstruct", examples, models, columns, suffixes);

// Generate tables for Control by Synthetic Input
examples = [
    {"id": "const-lo",   "label": "Constant: Low-Range RPM"},
    {"id": "const-mid",  "label": "Constant: Mid-Range RPM"},
    {"id": "const-hi",   "label": "Constant: High-Range RPM"},
    {"id": "ramp",       "label": "Varying: Ramp"},
    {"id": "osc-slow",   "label": "Varying: Slow Sine"},
    {"id": "osc-fast",   "label": "Varying: Fast Sine"},
    {"id": "outside-lo", "label": "Outside Dataset: Below"},
    {"id": "outside-hi", "label": "Outside Dataset: Above"}
];
models = ["vanilla", "mel", "cyl", "phase", "hnt", "large"];
columns = ["synthesized", "harmonic", "noise", "transients"];
suffixes = ["", "additive", "subtractive", "transients"];

generateTables("control", examples, models, columns, suffixes);