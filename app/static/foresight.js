function makeForesight(parentName, covid_data, state="New York", dataType='positive'){

  document.getElementById(parentName).innerHTML = ""
  // clears the data from the last time

  console.log(dataType)

  config = {
  pointType: 'mmwr-week', // Default is week
  axes: {
    y: {
      title: dataType // Title for the y axis
    },
    x: {
      title: 'Week of 2020'
    }
    },
  confidenceIntervals: ['50%', '75%'],
  }


  weekToEight = {}
  filtered = covid_data.filter(d => eightToDate(d.date).getDay() == 0)
  filtered = filtered.filter(d => d.state == states_hash[state])
  Object.entries(filtered).forEach(d => weekToEight[weekOfYear(eightToDate(d[1].date))] = d[1][dataType] )


  timePoints = [...Array(51).keys()].map(w => {
    return { week: w + 1, year: 2020 }
  })


  weekNow = weekOfYear(new Date())

  actual = []
  for (i = 0; i < 51; i++) {
    if (i in weekToEight){
      actual.push(weekToEight[i])
    }else{
      actual.push(null)
    }}

  // this is where the RL/simulations models comes in
  predictions = timePoints.map(tp => {
    if (tp.week >= weekNow) {
    // We only predict upto week 30
      return null
    } else {
      to_plot = []
      for (i = tp.week; i < weekNow; i++) {
        if (typeof weekToEight[i] == "number"){
          r = parseFloat(weekToEight[i])
          to_plot.push({point: r,
                        low: [Math.max(0, r - (r * .1)), Math.max(0, r - r * 0.2)],
                        high: [Math.max(0, r + (r * .1)), Math.max(0, r + (r * .2))]})
        } else {
          to_plot.push({point: parseFloat(0)})
        }
      }
      return {
        series: Array.from(to_plot)
      }
    }
  })

  data = {
    timePoints,
    actual: actual,
    models: [
      {
        id: 'actual tester',
        meta: {
          name: 'Name',
          description: 'Model description here',
          url: 'http://github.com'
        },
        pinned: false, // Setting true shows the model in top section of the legend
                       // In case of absence of `pinned` key (or false), the model
                       // goes in the bottom section
        predictions: predictions,
        style: { // Optional parameter for applying custom css on svg elements
          color: '#4682b4', // Defaults to values from the internal palette
          point: {
            // Style for the dots in prediction
          },
          area: {
            // Style for the confidence area (shaded region around the line)
          },
          line: {
            // Style for the main line
          }
        }
      }
    ]
  }

  timeChart = new d3Foresight.TimeChart('#' + parentName, config)
  timeChart.plot(data)
  timeChart.update(10)
  timeChart.moveForward()
  timeChart.moveBackward()

}
