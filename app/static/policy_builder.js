function tableToJson(id, type='index') {
  var table = document.getElementById(id)
  var data = [];
  // first row needs to be headers
  var headers = [];
  for (var i=0; i<table.rows[0].cells.length; i++) {
      headers[i] = table.rows[0].cells[i].innerHTML.replace(/ /gi,'');
  }
  if (type == 'index'){
    // go through cells
  	start = 1
    for (var i=start; i<table.rows.length; i++) {
        var tableRow = table.rows[i];
        var rowData = {};
        for (var j=0; j<tableRow.cells.length ; j++) {
            rowData[headers[j]] = tableRow.cells[j].innerHTML;
        }
        data.push(rowData);
    }
  }else{
    // go through cells
    for (var i=0; i<table.rows[0].cells.length; i++) {
      var colData = [];
      remember = parseFloat(0)
      for (var j=1; j<table.rows.length ; j++) {
        if (i==0){
          colData.push(parseFloat(table.rows[j].cells[i].innerHTML.split(" ")[1]));
        }else{
          if (table.rows[j].cells[i].innerHTML != ''){
            remember = table.rows[j].cells[i].innerHTML
            colData.push(parseFloat(table.rows[j].cells[i].innerHTML));
          } else {
            colData.push(parseFloat(remember));
          }
      }}
      data[headers[i]] = colData;
    }
  }
  return data;
}
