function chart(data, accessor_f) {
  const margin = ({top: 0, right: 0, bottom: 0, left: 20})

  const width = 1000
  const height = 1000
  const dx = 10
  const dy = 75

  const tree = d3.tree().nodeSize([dx, dy])
  const diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x)

  const firstchars = (x) => x.substring(0, 12)
  const exists = (i, x) => (i in x) && x[i].length

  const SELECTED_COLOR = "#090"
  const SELECTED_COLOR_NODE = "#0b0"

  function nodeColor(node){ 
    return node.data.selected ? SELECTED_COLOR_NODE : "#555"
  }

  const root = d3.hierarchy(data, accessor_f);

  root.x0 = dy / 2;
  root.y0 = 0;
  root.descendants().forEach((d, i) => {
    d.id = i;
    d._children = d.children;
    d.open = d.data.open;

    if(!d.open){
      d.children = null
    }
    // if (d.depth && d.data.name.length !== 7) d.children = null;
  });

  // console.log(root)

  const svg = d3.create("svg")

  const svg_group = svg.append("g")
      .attr("transform",
            `translate(85,350)`)

      // // .attr("viewBox", [-margin.left, -margin.top, width, dx])
      // .style("font", "10px sans-serif")
      // .style("user-select", "none");

  const gLink = svg_group.append("g")
      .attr("fill", "none")
      .attr("stroke", "#555")
      .attr("stroke-opacity", 1.0)
      .attr("stroke-width", .5);

  const gNode = svg_group.append("g")
      .attr("cursor", "pointer")
      .attr("pointer-events", "all");

  function update(source) {
    // const duration = d3.event && d3.event.altKey ? 2500 : 250;
    // const duration = 0;
    const nodes = root.descendants().reverse();
    const links = root.links();

    // Compute the new tree layout.
    tree(root);

    // console.log("update")

    let left = root;
    let right = root;
    root.eachBefore(node => {
      if (node.x < left.x) left = node;
      if (node.x > right.x) right = node;
    });

    const height = right.x - left.x + margin.top + margin.bottom;

    const transition = null //svg_group.transition()
        // .duration(duration)
        // .attr("viewBox // ", //  [-margin.left, left.x - margin.top, width, height])
        // .tween("resize", window.ResizeObserver ? null : () => () => svg_group.dispatch("toggle"));

    // Update the nodes…
    const node = gNode.selectAll("g")
      .data(nodes, d => d.id);

    // Enter any new nodes at the parent's previous position.
    const nodeEnter = node.enter().append("g");
    nodeEnter
        .attr("transform", d => `translate(${source.y0},${source.x0})`)
        .attr("fill-opacity", 0)
        .attr("stroke-opacity", 0)
        .on("click", d => {
              // d.children = null;
          d.open = !d.open;
          d.children = d.open ? d._children : null;
          update(d);
        });

    nodeEnter.append("circle")
        .attr("r", 2.5)

    // for easier mouse click
    nodeEnter.append("circle")
        .attr("r", 5)
        .attr("opacity", 0)

    nodeEnter.append("text")
        .attr("dy", "0.31em")
        .attr("x", d => d._children ? -10 : 6)
        .attr("text-anchor", d => d._children ? "end" : "start")
      .clone(true).lower()
        .attr("stroke-linejoin", "round")
        .attr("stroke-width", 3)
        .attr("stroke", "white");

    // Transition nodes to their new position.
    const nodeUpdate = node.merge(nodeEnter) // .transition(transition)

    function formatNumber(x){
      if (typeof(x) === 'number') {
        if(x % 1 === 0)
          return x
        else
          return x.toFixed(2)
      }
      else{
        return x;
      }
    }

    nodeUpdate.selectAll("text")
        .text(d => {
          if (d._children)
            return d.data.name ? d.data.name : "";
          else {
            if (d.open)
              return d.data.name + "=" + formatNumber(d.data.value);
            else 
              return d.data.name;
          }
        })

    nodeUpdate
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .attr("fill-opacity", 1)
        .attr("stroke-opacity", 1)

    nodeUpdate.select('circle')
        .attr("fill", d => d.open ? nodeColor(d) : "#fff")
        .attr("stroke", d => d.open ? "none" : nodeColor(d));

    // Transition exiting nodes to the parent's new position.
    const nodeExit = node.exit()
    nodeExit.remove()
        .attr("transform", d => `translate(${source.y},${source.x})`)
        .attr("fill-opacity", 0)
        .attr("stroke-opacity", 0);

    // Update the links…
    const link = gLink.selectAll("path")
      .data(links, d => d.target.id);

    // Enter any new links at the parent's previous position.
    const linkEnter = link.enter().append("path")
        .attr("d", d => {
          const o = {x: source.x0, y: source.y0};
          return diagonal({source: o, target: o});
        })
        .attr("stroke-width", d => d.target.data.prob == 0 ? .5 : Math.max(d.target.data.prob * 10, .5))
        // .attr("stroke-opacity", d => d.target.data.prob == 0 ? 1.0 : 0.4)
        // .attr("stroke-dasharray", d => d.target.data.prob == 0 ? 4 : 0)
        .attr("stroke-dasharray", d => !d.target.data.prob ? 4 : 0)
        .attr("stroke", d => d.target.data.selected ? SELECTED_COLOR : "#000")


    // Transition links to their new position.
    link.merge(linkEnter)//.transition(transition)
        .attr("d", diagonal);

    // Transition exiting nodes to the parent's new position.
    link.exit().remove()
        .attr("d", d => {
          const o = {x: source.x, y: source.y};
          return diagonal({source: o, target: o});
        });

    // Stash the old positions for transition.
    root.eachBefore(d => {
      d.x0 = d.x;
      d.y0 = d.y;
    });
  }

  update(root);
  return [svg_group.node(), root, update];
}