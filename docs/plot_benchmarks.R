#!/usr/bin/env Rscript
# Zero-dependency base-R (grid + cairo) renderer for docs/benchmark_runs.csv.
# Produces docs/vxfec_benchmarks.png without requiring CRAN / ggplot2.

library(grid)
library(grDevices)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else "docs/benchmark_runs.csv"
out_path <- if (length(args) >= 2) args[2] else "docs/vxfec_benchmarks.png"

if (!file.exists(csv_path)) {
  stop(sprintf("Input CSV '%s' not found; pass a benchmark CSV path to docs/plot_benchmarks.R <runs.csv> [out.png].", csv_path), call. = FALSE)
}
raw <- read.csv(csv_path, stringsAsFactors = FALSE)
font_fam <- "sans"

draw_triangle <- function(x_npc, y_npc, r_mm, col) {
  h <- r_mm * 1.15
  xs <- unit.c(
    unit(x_npc, "npc"),
    unit(x_npc, "npc") - unit(r_mm, "mm"),
    unit(x_npc, "npc") + unit(r_mm, "mm")
  )
  ys <- unit.c(
    unit(y_npc, "npc") + unit(h * 0.68, "mm"),
    unit(y_npc, "npc") - unit(h * 0.52, "mm"),
    unit(y_npc, "npc") - unit(h * 0.52, "mm")
  )
  grid.polygon(x = xs, y = ys, gp = gpar(fill = col, col = col))
}

draw_diamond <- function(x_npc, y_npc, r_mm, col) {
  xs <- unit.c(
    unit(x_npc, "npc"),
    unit(x_npc, "npc") + unit(r_mm, "mm"),
    unit(x_npc, "npc"),
    unit(x_npc, "npc") - unit(r_mm, "mm")
  )
  ys <- unit.c(
    unit(y_npc, "npc") + unit(r_mm, "mm"),
    unit(y_npc, "npc"),
    unit(y_npc, "npc") - unit(r_mm, "mm"),
    unit(y_npc, "npc")
  )
  grid.polygon(x = xs, y = ys, gp = gpar(fill = col, col = col))
}

draw_panel_gpu <- function(df) {
  pushViewport(viewport(x = 0, y = 0, width = 1, height = 1, just = c("left", "bottom")))

  grid.text(
    "A. Resident-VRAM Uniform-Mask Burst Recovery",
    x = unit(14, "pt"), y = unit(1, "npc") - unit(13, "pt"),
    just = c("left", "top"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 12.2, col = "#0f172a")
  )
  grid.text(
    "1.18-5.39 Tbps Protected Payload / Decode Time (RX 7900 XTX, 1.0-1.5 GiB GDDR6)",
    x = unit(14, "pt"), y = unit(1, "npc") - unit(30, "pt"),
    just = c("left", "top"),
    gp = gpar(fontfamily = font_fam, fontsize = 9.0, col = "#475569")
  )

  left_pt <- 62
  right_pt <- 16
  top_pt <- 52
  bot_pt <- 84
  pushViewport(viewport(
    x = unit(left_pt, "pt"),
    y = unit(bot_pt, "pt"),
    width = unit(1, "npc") - unit(left_pt + right_pt, "pt"),
    height = unit(1, "npc") - unit(top_pt + bot_pt, "pt"),
    just = c("left", "bottom"),
    clip = "off"
  ))

  xmin <- 1.0
  xmax <- 54.5
  ymin <- 0.0
  ymax <- 6.2
  sx <- function(v) (v - xmin) / (xmax - xmin)
  sy <- function(v) (v - ymin) / (ymax - ymin)

  for (yv in 0:6) {
    yn <- sy(yv)
    grid.lines(x = c(0, 1), y = c(yn, yn), gp = gpar(col = "#e2e8f0", lwd = 1))
    grid.text(
      sprintf("%d Tbps", yv),
      x = unit(-7, "pt"), y = unit(yn, "npc"), just = c("right", "center"),
      gp = gpar(fontfamily = font_fam, fontsize = 8.5, col = "#475569")
    )
  }
  x_breaks <- c(5, 10, 20, 30, 50)
  for (xv in x_breaks) {
    xn <- sx(xv)
    grid.lines(x = c(xn, xn), y = c(0, 1), gp = gpar(col = "#e2e8f0", lty = "dashed", lwd = 1))
    grid.text(
      sprintf("%d%%", xv),
      x = unit(xn, "npc"), y = unit(-10, "pt"), just = c("center", "top"),
      gp = gpar(fontfamily = font_fam, fontsize = 8.5, col = "#475569")
    )
  }

  y1 <- sy(1.0)
  grid.lines(x = c(0, 1), y = c(y1, y1), gp = gpar(col = "#64748b", lty = "dotted", lwd = 1.4))
  grid.text(
    "1.0 Tbps",
    x = unit(sx(38), "npc"), y = unit(y1, "npc") - unit(4, "pt"),
    just = c("left", "top"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 8.0, col = "#64748b")
  )

  grid.text(
    "Simulated Systematic Erasure Rate",
    x = unit(0.5, "npc"), y = unit(-26, "pt"), just = c("center", "top"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 9.4, col = "#1e293b")
  )
  grid.text(
    "Protected Payload / Decode Time",
    x = unit(-46, "pt"), y = unit(0.5, "npc"), rot = 90, just = c("center", "bottom"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 9.4, col = "#1e293b")
  )

  s_rq <- "Rateless (RFC 6330 K=256 +19 GF2 Slack)"
  s_mds <- "MDS 4-Bit RegTab + TILE_E=16 (K=128)"
  s_gf8 <- "Rateless Low-Slack (RFC 6330 K=256 +3 GF8)"

  col_rq <- "#0f766e"
  col_mds <- "#2563eb"
  col_gf8 <- "#d97706"

  d1 <- df[df$series == s_rq, ]
  d1 <- d1[order(d1$x_val), ]
  if (nrow(d1) > 0) {
    grid.lines(x = sx(d1$x_val), y = sy(d1$y_val), gp = gpar(col = col_rq, lwd = 2.4))
    for (i in seq_len(nrow(d1))) {
      xn <- sx(d1$x_val[i])
      yn <- sy(d1$y_val[i])
      grid.circle(x = unit(xn, "npc"), y = unit(yn, "npc"), r = unit(1.3, "mm"), gp = gpar(fill = col_rq, col = col_rq))
      hj <- if (d1$x_val[i] >= 50) 0.85 else if (d1$x_val[i] <= 5) 0.25 else 0.35
      grid.text(
        d1$label[i],
        x = unit(xn, "npc"), y = unit(yn, "npc") + unit(6, "pt"),
        just = c(hj, 0),
        gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 8.1, col = col_rq)
      )
    }
  }

  d2 <- df[df$series == s_mds, ]
  d2 <- d2[order(d2$x_val), ]
  if (nrow(d2) > 0) {
    grid.lines(x = sx(d2$x_val), y = sy(d2$y_val), gp = gpar(col = col_mds, lwd = 2.4))
    for (i in seq_len(nrow(d2))) {
      xn <- sx(d2$x_val[i])
      yn <- sy(d2$y_val[i])
      draw_triangle(xn, yn, 1.4, col_mds)
      dx_pt <- if (d2$x_val[i] <= 10) 5.0 else 0.0
      dy_pt <- if (d2$x_val[i] == 5) 3.8 else if (d2$x_val[i] == 10) 4.2 else 6.0
      hj <- if (d2$x_val[i] <= 10) 0.0 else 0.35
      grid.text(
        d2$label[i],
        x = unit(xn, "npc") + unit(dx_pt, "pt"), y = unit(yn, "npc") + unit(dy_pt, "pt"),
        just = c(hj, 0),
        gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 8.1, col = col_mds)
      )
    }
  }

  d3 <- df[df$series == s_gf8, ]
  if (nrow(d3) > 0) {
    for (i in seq_len(nrow(d3))) {
      xn <- sx(d3$x_val[i])
      yn <- sy(d3$y_val[i])
      draw_diamond(xn, yn, 1.5, col_gf8)
      grid.text(
        d3$label[i],
        x = unit(xn, "npc") + unit(5, "pt"), y = unit(yn, "npc") - unit(5.5, "pt"),
        just = c("left", "top"),
        gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 7.9, col = "#b45309")
      )
    }
  }

  leg_items <- list(
    list(lbl = s_rq, col = col_rq, shp = "circle"),
    list(lbl = s_mds, col = col_mds, shp = "tri"),
    list(lbl = s_gf8, col = col_gf8, shp = "dia")
  )
  for (k in seq_along(leg_items)) {
    ly <- unit(-42 - (k - 1) * 12.5, "pt")
    lx0 <- unit(0.05, "npc")
    lx1 <- unit(0.05, "npc") + unit(14, "pt")
    lxm <- unit(0.05, "npc") + unit(7, "pt")
    it <- leg_items[[k]]
    if (it$shp != "dia") {
      grid.lines(x = unit.c(lx0, lx1), y = unit.c(ly, ly), gp = gpar(col = it$col, lwd = 2.2))
    }
    if (it$shp == "circle") {
      grid.circle(x = lxm, y = ly, r = unit(1.15, "mm"), gp = gpar(fill = it$col, col = it$col))
    } else if (it$shp == "tri") {
      pushViewport(viewport(x = lxm, y = ly, width = unit(4, "mm"), height = unit(4, "mm")))
      draw_triangle(0.5, 0.5, 1.2, it$col)
      popViewport()
    } else {
      pushViewport(viewport(x = lxm, y = ly, width = unit(4, "mm"), height = unit(4, "mm")))
      draw_diamond(0.5, 0.5, 1.3, it$col)
      popViewport()
    }
    grid.text(
      it$lbl,
      x = lx1 + unit(6, "pt"), y = ly, just = c("left", "center"),
      gp = gpar(fontfamily = font_fam, fontsize = 8.2, col = "#1e293b")
    )
  }

  popViewport()
  popViewport()
}

draw_panel_pipe <- function(df) {
  pushViewport(viewport(x = 0, y = 0, width = 1, height = 1, just = c("left", "bottom")))

  grid.text(
    "B. Wire-Ring Superframe Pipeline (Resident VRAM)",
    x = unit(10, "pt"), y = unit(1, "npc") - unit(13, "pt"),
    just = c("left", "top"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 12.2, col = "#0f172a")
  )
  grid.text(
    "Avg Rate w/ 8 Poisoned Masks, Full-VRAM Verify & Copy Contention (Excludes NIC DMA)",
    x = unit(10, "pt"), y = unit(1, "npc") - unit(30, "pt"),
    just = c("left", "top"),
    gp = gpar(fontfamily = font_fam, fontsize = 9.0, col = "#475569")
  )

  left_pt <- 204
  right_pt <- 18
  top_pt <- 58
  bot_pt <- 60
  pushViewport(viewport(
    x = unit(left_pt, "pt"),
    y = unit(bot_pt, "pt"),
    width = unit(1, "npc") - unit(left_pt + right_pt, "pt"),
    height = unit(1, "npc") - unit(top_pt + bot_pt, "pt"),
    just = c("left", "bottom"),
    clip = "off"
  ))

  n <- nrow(df)
  xmax <- 2.75
  sx <- function(v) v / xmax
  sy <- function(i) (i - 0.5) / n

  for (xv in seq(0, 2.5, by = 0.5)) {
    xn <- sx(xv)
    grid.lines(x = c(xn, xn), y = c(0, 1), gp = gpar(col = "#e2e8f0", lwd = 1))
    grid.text(
      sprintf("%.1f", xv),
      x = unit(xn, "npc"), y = unit(-8, "pt"), just = c("center", "top"),
      gp = gpar(fontfamily = font_fam, fontsize = 8.5, col = "#475569")
    )
  }

  x1 <- sx(1.0)
  grid.lines(x = c(x1, x1), y = c(0, 1.03), gp = gpar(col = "#64748b", lty = "dotted", lwd = 1.4))
  grid.text(
    "1.0 Tbps Avg Rate",
    x = unit(x1, "npc") + unit(3, "pt"), y = unit(1.03, "npc"),
    just = c("left", "bottom"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 8.0, col = "#64748b")
  )

  bar_h <- 0.58 / n
  for (i in seq_len(n)) {
    yc <- sy(i)
    w <- sx(df$y_val[i])
    is_rq <- df$series[i] == "Rateless"
    bcol <- if (is_rq) "#0f766e" else "#2563eb"
    tcol <- if (is_rq) "#0f766e" else "#1e40af"

    grid.rect(
      x = unit(0, "npc"), y = unit(yc, "npc"),
      width = unit(w, "npc"), height = unit(bar_h, "npc"),
      just = c("left", "center"),
      gp = gpar(fill = bcol, col = NA)
    )
    grid.text(
      df$scenario[i],
      x = unit(-7, "pt"), y = unit(yc, "npc"), just = c("right", "center"),
      gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 8.5, col = "#1e293b")
    )
    grid.text(
      df$label[i],
      x = unit(w, "npc") + unit(5, "pt"), y = unit(yc, "npc"), just = c("left", "center"),
      gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 8.2, col = tcol)
    )
  }

  grid.text(
    "Average Protected Payload Processing Rate (Tbps)",
    x = unit(0.5, "npc"), y = unit(-23, "pt"), just = c("center", "top"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 9.4, col = "#1e293b")
  )

  ly <- unit(-42, "pt")
  grid.rect(x = unit(0.08, "npc"), y = ly, width = unit(9.5, "pt"), height = unit(9.5, "pt"),
            just = c("left", "center"), gp = gpar(fill = "#0f766e", col = NA))
  grid.text("Rateless (RFC 6330)", x = unit(0.08, "npc") + unit(13, "pt"), y = ly,
            just = c("left", "center"), gp = gpar(fontfamily = font_fam, fontsize = 8.5, col = "#1e293b"))

  grid.rect(x = unit(0.54, "npc"), y = ly, width = unit(9.5, "pt"), height = unit(9.5, "pt"),
            just = c("left", "center"), gp = gpar(fill = "#2563eb", col = NA))
  grid.text("MDS (Cauchy RS8 GF8)", x = unit(0.54, "npc") + unit(13, "pt"), y = ly,
            just = c("left", "center"), gp = gpar(fontfamily = font_fam, fontsize = 8.5, col = "#1e293b"))

  popViewport()
  popViewport()
}

draw_grouped_cct_panel <- function(df, title_txt, sub_txt, xmax, x_ticks, series_order,
                                   left_pt = 206, compact_legend = FALSE) {
  pushViewport(viewport(x = 0, y = 0, width = 1, height = 1, just = c("left", "bottom")))

  grid.text(
    title_txt,
    x = unit(14, "pt"), y = unit(1, "npc") - unit(11, "pt"),
    just = c("left", "top"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 12.2, col = "#0f172a")
  )
  grid.text(
    sub_txt,
    x = unit(14, "pt"), y = unit(1, "npc") - unit(28, "pt"),
    just = c("left", "top"),
    gp = gpar(fontfamily = font_fam, fontsize = 9.0, col = "#475569")
  )

  right_pt <- 22
  top_pt <- 46
  bot_pt <- 54
  pushViewport(viewport(
    x = unit(left_pt, "pt"),
    y = unit(bot_pt, "pt"),
    width = unit(1, "npc") - unit(left_pt + right_pt, "pt"),
    height = unit(1, "npc") - unit(top_pt + bot_pt, "pt"),
    just = c("left", "bottom"),
    clip = "off"
  ))

  scenarios <- unique(df$scenario)
  n <- length(scenarios)
  sx <- function(v) v / xmax
  sy <- function(i) (i - 0.5) / n

  for (xv in x_ticks) {
    xn <- sx(xv)
    grid.lines(x = c(xn, xn), y = c(0, 1), gp = gpar(col = "#e2e8f0", lwd = 1))
    grid.text(
      sprintf("%d", xv),
      x = unit(xn, "npc"), y = unit(-7, "pt"), just = c("center", "top"),
      gp = gpar(fontfamily = font_fam, fontsize = 8.5, col = "#475569")
    )
  }

  fill_vals <- c("#94a3b8", "#64748b", "#2563eb")
  txt_vals <- c("#475569", "#334155", "#0f766e")
  fill_map <- setNames(fill_vals, series_order)
  txt_map <- setNames(txt_vals, series_order)

  grp_span <- 0.78 / n
  for (i in seq_len(n)) {
    sc <- scenarios[i]
    sub <- df[df$scenario == sc, ]
    ord <- match(series_order, sub$series)
    ord <- ord[!is.na(ord)]
    sub <- sub[ord, , drop = FALSE]
    m <- nrow(sub)
    yc <- sy(i)

    if (i > 1) {
      ysep <- (i - 1) / n
      grid.lines(x = c(0, 1), y = c(ysep, ysep), gp = gpar(col = "#f1f5f9", lwd = 1))
    }

    grid.text(
      sc,
      x = unit(-8, "pt"), y = unit(yc, "npc"), just = c("right", "center"),
      gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 8.5, col = "#1e293b")
    )

    slot_h <- grp_span / m
    bar_h <- slot_h * 0.84
    for (j in seq_len(m)) {
      yj <- yc + ((m + 1) / 2 - j) * slot_h
      w <- sx(sub$y_val[j])
      sname <- sub$series[j]
      grid.rect(
        x = unit(0, "npc"), y = unit(yj, "npc"),
        width = unit(w, "npc"), height = unit(bar_h, "npc"),
        just = c("left", "center"),
        gp = gpar(fill = fill_map[[sname]], col = NA)
      )
      grid.text(
        sub$label[j],
        x = unit(w, "npc") + unit(4.5, "pt"), y = unit(yj, "npc"), just = c("left", "center"),
        gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 7.9, col = txt_map[[sname]])
      )
    }
  }

  grid.text(
    "Collective Completion Time (ms, lower is better)",
    x = unit(0.5, "npc"), y = unit(-21, "pt"), just = c("center", "top"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 9.2, col = "#1e293b")
  )

  popViewport()

  # Draw legend in outer panel viewport so half-width panels have full room
  ly <- unit(14, "pt")
  leg_labels <- if (compact_legend) {
    c("UDP Cumul.-NACK GBN", "RFC 2018 SACK (No FEC)", "vxfec (MDS / Rateless HARQ)")
  } else {
    sub("^User-Space UDP Go-Back-N$", "User-Space UDP Cumul.-NACK GBN", series_order)
  }
  leg_x <- if (compact_legend) c(0.06, 0.33, 0.65) else c(0.20, 0.46, 0.70)
  for (k in seq_along(series_order)) {
    sn <- series_order[k]
    grid.rect(
      x = unit(leg_x[k], "npc"), y = ly, width = unit(9, "pt"), height = unit(9, "pt"),
      just = c("left", "center"), gp = gpar(fill = fill_map[[sn]], col = NA)
    )
    grid.text(
      leg_labels[k],
      x = unit(leg_x[k], "npc") + unit(12.5, "pt"), y = ly,
      just = c("left", "center"), gp = gpar(fontfamily = font_fam, fontsize = 8.2, col = "#1e293b")
    )
  }

  popViewport()
}

draw_panel_cloud_wire <- function(df) {
  pushViewport(viewport(x = 0, y = 0, width = 1, height = 1, just = c("left", "bottom")))

  grid.text(
    "E. 200G TIER_1 Titanium gVNIC & Wire -> VRAM Throughput",
    x = unit(10, "pt"), y = unit(1, "npc") - unit(11, "pt"),
    just = c("left", "top"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 12.2, col = "#0f172a")
  )
  grid.text(
    "GCP 4x 100G gVNIC 8896B Jumbo (2x n2-standard-96 TIER_1), 32G & Local (Gbps)",
    x = unit(10, "pt"), y = unit(1, "npc") - unit(28, "pt"),
    just = c("left", "top"),
    gp = gpar(fontfamily = font_fam, fontsize = 9.0, col = "#475569")
  )

  # Full-panel-width callout badge for 24.4 ms Iowa <-> Virginia WAN Kernel-Drop AllReduce
  grid.roundrect(
    x = unit(10, "pt"), y = unit(1, "npc") - unit(50, "pt"),
    width = unit(1, "npc") - unit(22, "pt"), height = unit(16, "pt"),
    r = unit(3, "pt"), just = c("left", "center"),
    gp = gpar(fill = "#ecfdf5", col = "#0f766e", lwd = 1)
  )
  grid.text(
    "24.4 ms WAN Kernel-Drop (Iowa-VA): vxfec 49.8 ms p50 / 76.3 ms p99 (2.0x faster vs SACK 103.1 / 151.6 ms; GBN >3s)",
    x = unit(0.5, "npc"), y = unit(1, "npc") - unit(50, "pt"),
    just = c("center", "center"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 7.3, col = "#0f766e")
  )

  left_pt <- 192
  right_pt <- 14
  top_pt <- 66
  bot_pt <- 54
  pushViewport(viewport(
    x = unit(left_pt, "pt"),
    y = unit(bot_pt, "pt"),
    width = unit(1, "npc") - unit(left_pt + right_pt, "pt"),
    height = unit(1, "npc") - unit(top_pt + bot_pt, "pt"),
    just = c("left", "bottom"),
    clip = "off"
  ))

  n <- nrow(df)
  xmax <- 370.0
  sx <- function(v) v / xmax
  sy <- function(i) (i - 0.5) / n

  for (xv in c(0, 50, 100, 150, 200, 250)) {
    xn <- sx(xv)
    is_cap <- (xv == 100 || xv == 200)
    grid.lines(
      x = c(xn, xn), y = c(0, 1),
      gp = gpar(col = if (is_cap) "#d97706" else "#e2e8f0",
                lty = if (is_cap) "dashed" else "solid",
                lwd = if (is_cap) 1.3 else 1)
    )
    grid.text(
      sprintf("%d", xv),
      x = unit(xn, "npc"), y = unit(-7, "pt"), just = c("center", "top"),
      gp = gpar(fontfamily = font_fam, fontface = if (is_cap) "bold" else "plain",
                fontsize = 8.4, col = if (is_cap) "#b45309" else "#475569")
    )
  }

  bar_h <- 0.62 / n
  for (i in seq_len(n)) {
    yc <- sy(i)
    w <- sx(df$y_val[i])
    is_gcp <- grepl("GCP", df$series[i])
    bcol <- if (is_gcp) "#0f766e" else "#2563eb"
    tcol <- if (is_gcp) "#0f766e" else "#1e40af"

    grid.rect(
      x = unit(0, "npc"), y = unit(yc, "npc"),
      width = unit(w, "npc"), height = unit(bar_h, "npc"),
      just = c("left", "center"),
      gp = gpar(fill = bcol, col = NA)
    )
    grid.text(
      df$scenario[i],
      x = unit(-7, "pt"), y = unit(yc, "npc"), just = c("right", "center"),
      gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 8.1, col = "#1e293b")
    )
    grid.text(
      df$label[i],
      x = unit(w, "npc") + unit(4.5, "pt"), y = unit(yc, "npc"), just = c("left", "center"),
      gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 7.7, col = tcol)
    )
  }

  grid.text(
    "Multi-Rail Wire Throughput (Gbps; dashed = 100G VM / 200G BiDir Cap)",
    x = unit(0.46, "npc"), y = unit(-21, "pt"), just = c("center", "top"),
    gp = gpar(fontfamily = font_fam, fontface = "bold", fontsize = 8.9, col = "#1e293b")
  )

  popViewport()

  ly <- unit(14, "pt")
  grid.rect(x = unit(0.06, "npc"), y = ly, width = unit(9, "pt"), height = unit(9, "pt"),
            just = c("left", "center"), gp = gpar(fill = "#0f766e", col = NA))
  grid.text("GCP 4x 100G TIER_1 & 2x gVNIC (8896B Jumbo)", x = unit(0.06, "npc") + unit(12.5, "pt"), y = ly,
            just = c("left", "center"), gp = gpar(fontfamily = font_fam, fontsize = 8.2, col = "#1e293b"))

  grid.rect(x = unit(0.60, "npc"), y = ly, width = unit(9, "pt"), height = unit(9, "pt"),
            just = c("left", "center"), gp = gpar(fill = "#2563eb", col = NA))
  grid.text("Local 4-Rail + RX 7900 XTX", x = unit(0.60, "npc") + unit(12.5, "pt"), y = ly,
            just = c("left", "center"), gp = gpar(fontfamily = font_fam, fontsize = 8.2, col = "#1e293b"))

  popViewport()
}

png(filename = out_path, width = 1600, height = 1380, res = 125, type = "cairo")
grid.newpage()
grid.rect(gp = gpar(fill = "#ffffff", col = NA))

pushViewport(viewport(layout = grid.layout(
  nrow = 3, ncol = 2,
  heights = unit(c(0.35, 0.31, 0.34), "npc"),
  widths = unit(c(0.49, 0.51), "npc")
)))

pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1))
draw_panel_gpu(subset(raw, panel == "gpu"))
popViewport()

pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 2))
draw_panel_pipe(subset(raw, panel == "pipe"))
popViewport()

pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 1:2))
draw_grouped_cct_panel(
  subset(raw, panel == "nccl"),
  "C. Single-Host 0-RTT Loopback Collective Completion Time (CCT)",
  "bench_nccl p99 (N=100) & Live 2-Process ROCm/RCCL Avg CCT vs. User-Space UDP Cumul.-NACK GBN & SACK",
  xmax = 9.2,
  x_ticks = 0:8,
  series_order = c("User-Space UDP Go-Back-N", "RFC 2018 SACK (No FEC)", "vxfec (MDS + HARQ + ECN)"),
  left_pt = 206
)
popViewport()

pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 1))
draw_grouped_cct_panel(
  subset(raw, panel == "cloud_cct"),
  "D. GCP Multi-Node 4x 100G gVNIC & Soft-RoCE (rdma_rxe) p99 CCT",
  "Collocated ToR (81us), 4x 100G Spine, Cross-Zone Drop (670us) & Soft-RoCE rdma_rxe UC",
  xmax = 33.0,
  x_ticks = c(0, 5, 10, 15, 20, 25),
  series_order = c("User-Space UDP Go-Back-N", "RFC 2018 SACK (No FEC)", "vxfec (MDS / Rateless HARQ)"),
  left_pt = 196,
  compact_legend = TRUE
)
popViewport()

pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 2))
draw_panel_cloud_wire(subset(raw, panel == "cloud_wire"))
popViewport()

popViewport()
dev.off()
cat("Wrote", out_path, "from", csv_path, "\n")
