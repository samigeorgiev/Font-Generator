import React from 'react';

import Toolbar from 'components/Navigation/Toolbar';

import styles from './index.module.css';

const layout = props => {
    return (
        <div className={styles.Layout}>
            <header>
                <Toolbar />
            </header>
            <main>
                {props.children}
            </main>
            <footer>
                <p>&copy; 2019 - {new Date().getFullYear()} All rights reserved | Sami</p>
            </footer>
        </div>
    );
};

export default layout;