bar_chart = function(data, true_y) {
  // set the dimensions and margins of the graph
  var margin = {top: 5, right: 20, bottom: 40, left: 40},
      width = 90 - margin.left - margin.right,
      height = 80 - margin.top - margin.bottom;

  // append the svg object to the body of the page
  var svg = d3
    .create("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      // .style("font", "10px sans-serif")

  svg_group = svg.append("g")
      .attr("transform",
            "translate(" + margin.left + "," + (margin.top + 330) + ")");


    // Add X axis
    var max_p = Math.max(...data);
    var max_shown = max_p >= 0.6 ? 1.0 : 0.6;
    var tick_values = max_p >= 0.6 ? [0., 1.0] : [0., 0.6];

    var x = d3.scaleLinear()
      .domain([0, max_shown])
      .range([ 0, width]);

    // Y axis
    var y = d3.scaleBand()
      .range([ 0, height ])
      .domain(d3.range(data.length))
      .padding(.1);


    //Bars
    svg_group.selectAll("myRect")
      .data(data)
      .enter()
      .append("rect")
      .attr("x", x(0) )
      .attr("y", (d, i) => y(i))
      .attr("width", d => x(d))
      .attr("height", y.bandwidth() )
      .attr("fill", (d, i) => true_y == i ? "black" : "gray")

    svg_group.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x).tickValues(tick_values).tickFormat(d3.format(".1f")))
      .attr("font-size", null)
      .attr("font-family", null)


    svg_group.append("g")
      .call(d3.axisLeft(y).tickFormat(i => (data.length > 4 ? "" : "cls_" + i)).tickSizeOuter(0))
      .attr("font-size", null)
      .attr("font-family", null)

    svg_group.append("text")
      .attr("transform", `translate(35 , ${height })`)
      .text("P(cls)");

  return svg_group.node()
}