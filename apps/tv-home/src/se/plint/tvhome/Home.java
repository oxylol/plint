package se.plint.tvhome;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.text.TextUtils;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

/**
 * The stick's home screen: YouTube, Stremio, and under MORE whatever else is on it, with Settings.
 * Plint Terminal look (theme/plint.css): one dark ground, one light, chamfered blocks, inverse video
 * for the block in focus. Real text views, so `plint tv screen` can read it.
 */
public class Home extends Activity {
    static final int BG = 0xff0f0e11, INK = 0xffe4f8ea, FAINT = 0x47e4f8ea, GLOW = 0x61e4f8ea;
    static final String[][] TILES = {{"YOUTUBE", "com.google.android.youtube.tv"}, {"STREMIO", "com.stremio.one"}};
    // the mark: a plinth and two stones, each cut top-left and bottom-right (x0, y0, x1, y1, cut in a 128x88 box)
    static final int[][] STONES = {{6, 54, 122, 82, 8}, {20, 30, 98, 48, 6}, {44, 8, 106, 24, 5}};

    float dp;
    Typeface face;
    View front, list, first, moreButton;
    LinearLayout rows;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        dp = getResources().getDisplayMetrics().density;
        face = new Typeface.Builder(getAssets(), "GeistMono.ttf").setFontVariationSettings("'wght' 600").build();
        getWindow().setBackgroundDrawable(new ColorDrawable(BG));

        LinearLayout tiles = new LinearLayout(this);
        tiles.setGravity(Gravity.CENTER);
        List<View> lit = new ArrayList<>();
        for (String[] t : TILES) {
            Intent go = launch(t[1]);
            TextView tile = block(t[0], 30, 9, go == null ? null : () -> startActivity(go));
            tile.setGravity(Gravity.CENTER);
            tiles.addView(tile, margins(px(300), px(190), px(14), 0));
            if (first == null && go != null) first = tile;
            lit.add(tile);
        }
        moreButton = block("MORE", 14, 6, this::showMore);
        ((TextView) moreButton).setGravity(Gravity.CENTER);
        if (first == null) first = moreButton;
        lit.add(moreButton);

        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setGravity(Gravity.CENTER);
        col.addView(tiles);
        col.addView(moreButton, margins(px(140), px(44), 0, px(40)));
        front = col;

        rows = new LinearLayout(this);
        rows.setOrientation(LinearLayout.VERTICAL);
        rows.setPadding(0, px(48), 0, px(48));
        ScrollView scroll = new ScrollView(this);
        scroll.addView(rows, new FrameLayout.LayoutParams(px(460), -2, Gravity.CENTER_HORIZONTAL));
        scroll.setVisibility(View.GONE);
        list = scroll;

        View mark = new Mark();
        lit.add(0, mark);
        FrameLayout root = new FrameLayout(this);
        root.addView(front, new FrameLayout.LayoutParams(-1, -1));
        root.addView(list, new FrameLayout.LayoutParams(-1, -1));
        FrameLayout.LayoutParams corner = new FrameLayout.LayoutParams(px(64), px(44), Gravity.TOP | Gravity.LEFT);
        corner.setMargins(px(52), px(40), 0, 0);   // clear of an old TV's overscan
        root.addView(mark, corner);
        setContentView(root);
        first.requestFocus();

        // a cold start switches the parts on one after another, each with a stutter; never eased
        Handler h = new Handler();
        for (int i = 0; i < lit.size(); i++) {
            View v = lit.get(i);
            float full = v.isFocusable() || v instanceof Mark ? 1f : .35f;
            v.setAlpha(0);
            h.postDelayed(() -> v.setAlpha(full), 90 + i * 130);
            h.postDelayed(() -> v.setAlpha(.3f * full), 140 + i * 130);
            h.postDelayed(() -> v.setAlpha(full), 200 + i * 130);
        }
    }

    /** The Home button while this is already in front: back to the two tiles. */
    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        showFront(first);
    }

    @Override public void onBackPressed() {
        if (list.getVisibility() == View.VISIBLE) showFront(moreButton);
    }

    void showFront(View focus) {
        list.setVisibility(View.GONE);
        front.setVisibility(View.VISIBLE);
        focus.requestFocus();
    }

    /** Settings, then every other TV app by name, and any app he has put on himself that is not made for a TV. */
    void showMore() {
        rows.removeAllViews();
        TreeMap<String, Intent> apps = new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
        PackageManager pm = getPackageManager();
        for (String cat : new String[] {Intent.CATEGORY_LAUNCHER, Intent.CATEGORY_LEANBACK_LAUNCHER}) {
            for (ResolveInfo r : pm.queryIntentActivities(new Intent(Intent.ACTION_MAIN).addCategory(cat), 0)) {
                String pkg = r.activityInfo.packageName;
                boolean builtIn = (r.activityInfo.applicationInfo.flags & ApplicationInfo.FLAG_SYSTEM) != 0;
                if (cat.equals(Intent.CATEGORY_LAUNCHER) && builtIn) continue;   // the system's odds and ends
                if (pkg.equals(getPackageName()) || pkg.equals(TILES[0][1]) || pkg.equals(TILES[1][1])) continue;
                Intent go = launch(pkg);
                if (go != null) apps.put(r.loadLabel(pm).toString().toUpperCase(), go);
            }
        }
        View top = row("SETTINGS", new Intent(Settings.ACTION_SETTINGS));
        for (String name : apps.keySet()) row(name, apps.get(name));
        front.setVisibility(View.GONE);
        list.setVisibility(View.VISIBLE);
        top.requestFocus();
    }

    View row(String name, Intent go) {
        TextView r = block(name, 16, 6, () -> startActivity(go));
        r.setGravity(Gravity.CENTER_VERTICAL);
        r.setPadding(px(18), 0, px(18), 0);
        r.setSingleLine();
        r.setEllipsize(TextUtils.TruncateAt.END);
        rows.addView(r, margins(-1, px(46), 0, px(8)));
        return r;
    }

    Intent launch(String pkg) {
        PackageManager pm = getPackageManager();
        Intent go = pm.getLeanbackLaunchIntentForPackage(pkg);
        return go != null ? go : pm.getLaunchIntentForPackage(pkg);
    }

    /** A chamfered block with a label. With nothing to do (the app is not on the stick) it stays dim and out of reach. */
    TextView block(String label, float size, float cut, Runnable go) {
        TextView t = new TextView(this);
        Block b = new Block(cut * dp);
        t.setText(label);
        t.setTypeface(face);
        t.setTextSize(TypedValue.COMPLEX_UNIT_DIP, size);
        t.setLetterSpacing(.14f);
        t.setTextColor(INK);
        t.setShadowLayer(9, 0, 0, GLOW);
        t.setBackground(b);
        if (go == null) {
            t.setAlpha(.35f);
            return t;
        }
        t.setFocusable(true);
        t.setDefaultFocusHighlightEnabled(false);   // the block draws its own
        t.setOnClickListener(v -> go.run());
        t.setOnFocusChangeListener((v, has) -> {
            b.lit = has;
            t.setTextColor(has ? BG : INK);
            t.setShadowLayer(has ? 0 : 9, 0, 0, GLOW);
            t.invalidate();
        });
        return t;
    }

    int px(float d) { return Math.round(d * dp); }

    LinearLayout.LayoutParams margins(int w, int h, int side, int top) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(w, h);
        p.setMargins(side, top, side, 0);
        return p;
    }

    static Path cut(float x0, float y0, float x1, float y1, float c) {
        Path p = new Path();
        p.moveTo(x0 + c, y0); p.lineTo(x1, y0); p.lineTo(x1, y1 - c);
        p.lineTo(x1 - c, y1); p.lineTo(x0, y1); p.lineTo(x0, y0 + c);
        p.close();
        return p;
    }

    /** A hairline frame; solid when it is the one in focus (inverse video). */
    static class Block extends Drawable {
        final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        final float c;
        boolean lit;

        Block(float c) { this.c = c; }

        @Override public void draw(Canvas canvas) {
            Rect b = getBounds();
            float in = lit ? 0 : 1;
            paint.setStyle(lit ? Paint.Style.FILL : Paint.Style.STROKE);
            paint.setStrokeWidth(2);
            paint.setColor(lit ? INK : FAINT);
            canvas.drawPath(cut(b.left + in, b.top + in, b.right - in, b.bottom - in, c), paint);
        }

        @Override public void setAlpha(int a) { }
        @Override public void setColorFilter(ColorFilter f) { }
        @Override public int getOpacity() { return PixelFormat.TRANSLUCENT; }
    }

    class Mark extends View {
        final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

        Mark() { super(Home.this); paint.setColor(INK); }

        @Override protected void onDraw(Canvas canvas) {
            float k = getWidth() / 128f;
            for (int[] s : STONES) canvas.drawPath(cut(s[0] * k, s[1] * k, s[2] * k, s[3] * k, s[4] * k), paint);
        }
    }
}
